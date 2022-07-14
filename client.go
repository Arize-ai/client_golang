package arize

import (
	"bytes"
	"context"
	"fmt"
	"io/ioutil"
	"net/http"
	"strconv"
	"time"

	pb "github.com/Arize-ai/client_golang/receiver/protocol/public"

	"google.golang.org/protobuf/types/known/wrapperspb"

	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/types/known/timestamppb"

	"github.com/pkg/errors"
)

const (
	baseURL    = "https://api.arize.com"
	sdkVersion = "0.1.0"
)

// Client for Arize api.
type Client struct {
	client   httpClient
	baseURL  string
	spaceKey string
	apiKey   string
	headers  map[string][]string
}

type httpClient interface {
	Do(req *http.Request) (*http.Response, error)
}

func NewClient(spaceKey, apiKey string) *Client {
	heads := map[string][]string{
		"authorization":             {apiKey},
		"Grpc-Metadata-space":       {spaceKey},
		"Grpc-Metadata-sdk-version": {sdkVersion},
		"Grpc-Metadata-sdk":         {"go"},
	}
	return &Client{
		client:   http.DefaultClient,
		apiKey:   apiKey,
		spaceKey: spaceKey,
		baseURL:  baseURL,
		headers:  heads,
	}
}

func (c *Client) Log(ctx context.Context, modelID string, modelVersion *string, predictionID string, features, tags map[string]interface{}, shapValues map[string]float64, prediction, actual interface{}, timestamp *time.Time, embeddingFeatures map[string]*Embeddings) (*Response, error) {
	if modelID == "" {
		return nil, errors.New("modelID can not be empty")
	}
	if predictionID == "" {
		return nil, errors.New("predictionID can not be empty")
	}

	predictionLabel, err := parseLabel(prediction)
	if err != nil {
		return nil, err
	}

	actualLabel, err := parseLabel(actual)
	if err != nil {
		return nil, err
	}

	if err = validateLabels(predictionLabel, actualLabel); err != nil {
		return nil, err
	}

	var ts *timestamppb.Timestamp
	if timestamp == nil {
		ts = timestamppb.Now()
	} else {
		ts = timestamppb.New(*timestamp)
	}

	tagsM, err := parseDimensions(tags)
	if err != nil {
		return nil, err
	}

	rec := &pb.Record{
		SpaceKey:     c.spaceKey,
		ModelId:      modelID,
		PredictionId: predictionID,
	}

	var mv string
	if modelVersion != nil {
		mv = *modelVersion
	}

	// set prediction message if applicable
	if predictionLabel != nil {
		fs, err := parseDimensions(features)
		if err != nil {
			return nil, err
		}
		es, err := parseEmbeddings(embeddingFeatures)
		if err != nil {
			return nil, err
		}
		for k, v := range es {
			fs[k] = v
		}
		pred := &pb.Prediction{
			Timestamp:    ts,
			ModelVersion: mv,
			Label:        predictionLabel,
			Features:     fs,
			Tags:         tagsM,
		}
		rec.Prediction = pred
	}

	// set actual label if applicable
	if actualLabel != nil {
		a := &pb.Actual{
			Label:     actualLabel,
			Tags:      tagsM,
			Timestamp: ts,
		}
		rec.Actual = a
	}

	if shapValues != nil && len(shapValues) > 0 {
		fi := &pb.FeatureImportances{
			FeatureImportances: shapValues,
			ModelVersion:       mv,
			Timestamp:          ts,
		}
		rec.FeatureImportances = fi
	}

	body, err := protojson.Marshal(rec)
	if err != nil {
		return nil, fmt.Errorf("error creating request body: %w", err)
	}

	uri := fmt.Sprintf("%s/v1/%s", c.baseURL, "log")
	req, err := http.NewRequestWithContext(ctx, "POST", uri, bytes.NewBuffer(body))
	if err != nil {
		return nil, errors.Wrap(err, "error creating new request to arize")
	}
	req.Header = c.headers

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, errors.Wrap(err, "HTTP request failure on request to arize")
	}
	defer resp.Body.Close()

	b, _ := ioutil.ReadAll(resp.Body)
	return &Response{StatusCode: resp.StatusCode, Body: string(b)}, nil
}

func validateLabels(pl, al *pb.Label) error {
	// this validation is only be evaluated if both labels are non-nil
	if pl == nil || al == nil {
		return nil
	}
	switch pl.GetData().(type) {
	case *pb.Label_ScoreCategorical:
		if _, ok := al.GetData().(*pb.Label_ScoreCategorical); !ok {
			return errors.Errorf("prediction and actual labels need to be the same type. prediction = %v, actual = %v", pl, al)
		}
	case *pb.Label_Numeric:
		if _, ok := al.GetData().(*pb.Label_Numeric); !ok {
			return errors.Errorf("prediction and actual labels need to be the same type. prediction = %v, actual = %v", pl, al)
		}
	default:
		return errors.Errorf("unknown prediciton label. prediction = %v", pl)
	}
	return nil
}

func parseLabel(labelValue interface{}) (*pb.Label, error) {
	if labelValue == nil {
		return nil, nil
	}

	switch labelValue.(type) {
	case *ScoreCategorical:
		lv := labelValue.(*ScoreCategorical)
		st := &pb.ScoreCategorical_ScoreCategory_{ScoreCategory: &pb.ScoreCategorical_ScoreCategory{Category: lv.Category, Score: lv.Score, NumericSequence: lv.NumericSequence}}
		scc := &pb.ScoreCategorical{Type: st}
		return &pb.Label{Data: &pb.Label_ScoreCategorical{ScoreCategorical: scc}}, nil
	case string:
		st := &pb.ScoreCategorical_Category_{Category: &pb.ScoreCategorical_Category{Category: labelValue.(string)}}
		scc := &pb.ScoreCategorical{Type: st}
		return &pb.Label{Data: &pb.Label_ScoreCategorical{ScoreCategorical: scc}}, nil
	case bool:
		st := &pb.ScoreCategorical_Category_{Category: &pb.ScoreCategorical_Category{Category: strconv.FormatBool(labelValue.(bool))}}
		scc := &pb.ScoreCategorical{Type: st}
		return &pb.Label{Data: &pb.Label_ScoreCategorical{ScoreCategorical: scc}}, nil
	case int:
		return &pb.Label{Data: &pb.Label_Numeric{Numeric: float64(labelValue.(int))}}, nil
	case int64:
		return &pb.Label{Data: &pb.Label_Numeric{Numeric: float64(labelValue.(int64))}}, nil
	case int32:
		return &pb.Label{Data: &pb.Label_Numeric{Numeric: float64(labelValue.(int32))}}, nil
	case float64:
		return &pb.Label{Data: &pb.Label_Numeric{Numeric: labelValue.(float64)}}, nil
	case float32:
		return &pb.Label{Data: &pb.Label_Numeric{Numeric: float64(labelValue.(float32))}}, nil
	default:
		return nil, errors.Errorf("unknown type for labelValue = %v", labelValue)
	}
}

func parseDimensions(dims map[string]interface{}) (map[string]*pb.Value, error) {
	dms := make(map[string]*pb.Value)
	for k, v := range dims {
		pv, err := parseValue(v)
		if err != nil {
			return nil, err
		}
		dms[k] = pv
	}
	return dms, nil
}

func parseValue(val interface{}) (*pb.Value, error) {
	switch val.(type) {
	case string:
		return &pb.Value{Data: &pb.Value_String_{String_: val.(string)}}, nil
	case bool:
		return &pb.Value{Data: &pb.Value_String_{String_: strconv.FormatBool(val.(bool))}}, nil
	case float64:
		return &pb.Value{Data: &pb.Value_Double{Double: val.(float64)}}, nil
	case float32:
		return &pb.Value{Data: &pb.Value_Double{Double: float64(val.(float32))}}, nil
	case int:
		return &pb.Value{Data: &pb.Value_Int{Int: int64(val.(int))}}, nil
	case int32:
		return &pb.Value{Data: &pb.Value_Int{Int: int64(val.(int32))}}, nil
	case int64:
		return &pb.Value{Data: &pb.Value_Int{Int: val.(int64)}}, nil
	default:
		return nil, errors.Errorf("unknown type for value = %v ", val)
	}
}

func parseEmbeddings(es map[string]*Embeddings) (map[string]*pb.Value, error) {
	ret := make(map[string]*pb.Value, len(es))
	for k, e := range es {
		if len(e.Vector) == 0 {
			return nil, errors.Errorf("embeddings.vector can not be empty")
		}
		emb := &pb.Embedding{}
		emb.Vector = e.Vector
		if e.LinkToData != nil {
			emb.LinkToData = wrapperspb.String(*e.LinkToData)
		}
		tok := &pb.Embedding_TokenArray{}
		for _, v := range e.Data {
			tok.Tokens = append(tok.Tokens, v)
		}
		rd := &pb.Embedding_RawData_TokenArray{TokenArray: tok}
		raw := &pb.Embedding_RawData{Type: rd}
		emb.RawData = raw

		ret[k] = &pb.Value{Data: &pb.Value_Embedding{Embedding: emb}}
	}
	return ret, nil
}

type ScoreCategorical struct {
	Score           float64
	Category        string
	NumericSequence []float64
}

type Embeddings struct {
	Vector     []float64
	Data       []string
	LinkToData *string
}

type Response struct {
	StatusCode int
	Body       string
}
