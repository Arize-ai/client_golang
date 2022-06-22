package arize

import (
	"bytes"
	"context"
	"fmt"
	"io/ioutil"
	"net/http"
	"testing"
	"time"

	pb "github.com/Arize-ai/client_golang/receiver/protocol/public"

	"github.com/stretchr/testify/assert"
	"google.golang.org/protobuf/encoding/protojson"
)

func TestFullLog(t *testing.T) {
	c := testClient()
	features := map[string]interface{}{
		"x": 1,
		"y": "y1",
		"z": 1.33,
	}
	tags := map[string]interface{}{
		"t": 1,
	}
	shap := map[string]float64{
		"s": 1.76,
	}
	dl := "gs://my/thing"
	e := &Embeddings{
		Vector:     []float64{1.34, 5.67},
		LinkToData: &dl,
		Data:       []string{"harrison wuz here"},
	}
	es := map[string]*Embeddings{
		"e": e,
	}
	ts := time.Now()
	mv := "test_model_version"
	resp, err := c.Log(context.Background(), "sdk_test_model", &mv, "test_pred_id", features, tags, shap, 0.65, 0.3, &ts, es)
	assert.NoError(t, err)
	assert.Equal(t, resp.StatusCode, 200)

	actualReq := c.client.(*mockHTTPClient).capturedRequests[0]
	bs, err := ioutil.ReadAll(actualReq.Body)
	if !assert.NoError(t, err) {
		t.FailNow()
	}
	rec := &pb.Record{}
	err = protojson.Unmarshal(bs, rec)
	if !assert.NoError(t, err) {
		t.FailNow()
	}

	assert.Equal(t, "sdk_test_model", rec.ModelId)
	assert.Equal(t, "test_model_version", rec.FeatureImportances.ModelVersion)
	assert.Equal(t, "test_pred_id", rec.PredictionId)

	actualActual := rec.Actual
	if !assert.NotNil(t, actualActual) {
		t.FailNow()
	}
	assert.Equal(t, actualActual.GetLabel().GetNumeric(), 0.3)
	assert.Equal(t, ts.Unix(), actualActual.Timestamp.Seconds)
	assert.Equal(t, 1, len(actualActual.Tags))
	assert.Equal(t, int64(1), actualActual.Tags["t"].GetInt())

	actualPred := rec.Prediction
	if !assert.NotNil(t, actualActual) {
		t.FailNow()
	}
	assert.Equal(t, "test_model_version", actualPred.ModelVersion)
	assert.Equal(t, ts.Unix(), actualPred.Timestamp.Seconds)
	assert.Equal(t, 0.65, actualPred.GetLabel().GetNumeric())
	assert.Equal(t, 1, len(actualPred.Tags))
	assert.Equal(t, int64(1), actualPred.Tags["t"].GetInt())
	assert.Equal(t, 4, len(actualPred.Features))
	assert.Equal(t, int64(1), actualPred.Features["x"].GetInt())
	assert.Equal(t, "y1", actualPred.Features["y"].GetString_())
	assert.Equal(t, 1.33, actualPred.Features["z"].GetDouble())

	actualEmbedding := actualPred.Features["e"].GetEmbedding()
	if !assert.NotNil(t, actualEmbedding) {
		t.FailNow()
	}
	assert.Equal(t, []float64{1.34, 5.67}, actualEmbedding.Vector)
	assert.Equal(t, "gs://my/thing", actualEmbedding.LinkToData.GetValue())
	assert.Equal(t, 1, len(actualEmbedding.RawData.GetTokenArray().GetTokens()))
	assert.Equal(t, "harrison wuz here", actualEmbedding.RawData.GetTokenArray().GetTokens()[0])

	actualShap := rec.FeatureImportances
	if !assert.NotNil(t, actualShap) {
		t.FailNow()
	}
	assert.Equal(t, 1, len(actualShap.FeatureImportances))
	assert.Equal(t, 1.76, actualShap.FeatureImportances["s"])
}

func TestLog(t *testing.T) {
	c := testClient()
	for i, tCase := range []struct {
		modelId      string
		predictionId string
		features     map[string]interface{}
		tags         map[string]interface{}
		prediction   interface{}
		actual       interface{}
		errExpected  bool
	}{
		{
			// modelId can not be an empty string
			modelId:      "",
			predictionId: "xyz",
			errExpected:  true,
		},
		{
			// prediction Id can not be an empty string
			modelId:      "mk",
			predictionId: "",
			errExpected:  true,
		},
		{
			// TODO(harrison): this probably shouldn't be valid since we enforce one of prediction / actual / featureimportance to be non nil in validation
			modelId:      "mk",
			predictionId: "xyz",
			errExpected:  false,
		},
		{
			modelId:      "mk",
			predictionId: "xyz",
			prediction:   0.5,
			errExpected:  false,
		},
		{
			// features can only be supported types
			modelId:      "mk",
			predictionId: "xyz",
			features:     map[string]interface{}{"x": uint64(2)},
			prediction:   0.5,
			errExpected:  true,
		},
		{
			// features do not support embeddings, embeddings have a separate embeddingFeatures argument
			modelId:      "mk",
			predictionId: "xyz",
			features: map[string]interface{}{"x": &Embeddings{
				Vector: []float64{1.0},
			}},
			prediction:  0.5,
			errExpected: true,
		},
		{
			// tags can only be supported types
			modelId:      "mk",
			predictionId: "xyz",
			tags:         map[string]interface{}{"x": uint64(2)},
			errExpected:  true,
		},
		{
			// tags cannot support embeddings only be supported types
			modelId:      "mk",
			predictionId: "xyz",
			tags: map[string]interface{}{"x": &Embeddings{
				Vector: []float64{1.0},
			}},
			errExpected: true,
		},
		{
			// prediction and actual must both by numeric or string types
			modelId:      "mk",
			predictionId: "xyz",
			prediction:   0.65,
			actual:       "bad",
			errExpected:  true,
		},
		{
			// prediction and actual can have different numeric types
			modelId:      "mk",
			predictionId: "xyz",
			prediction:   float64(0.65),
			actual:       int64(1),
			errExpected:  false,
		},
		{
			// prediction and actual can both be strings
			modelId:      "mk",
			predictionId: "xyz",
			prediction:   "hotdog",
			actual:       "notHotdog",
			errExpected:  false,
		},
		{
			// predictions and actuals can not be embeddings
			modelId:      "mk",
			predictionId: "xyz",
			prediction: &Embeddings{
				Vector: []float64{1.0},
			},
			actual: &Embeddings{
				Vector: []float64{0.5},
			},
			errExpected: true,
		},
	} {
		_, err := c.Log(context.Background(), tCase.modelId, nil, tCase.predictionId, tCase.features, tCase.tags, nil, tCase.prediction, tCase.actual, nil, nil)
		if tCase.errExpected {
			assert.Error(t, err, fmt.Sprintf("Log should return error for test case = %d", i))
		} else {
			assert.NoError(t, err, fmt.Sprintf("Log should not return error for test case = %d", i))
		}
	}
}

type mockHTTPClient struct {
	capturedRequests []*http.Request
	response         *http.Response
	err              error
}

func (m *mockHTTPClient) Do(req *http.Request) (*http.Response, error) {
	if m.capturedRequests == nil {
		m.capturedRequests = make([]*http.Request, 0, 1)
	}
	m.capturedRequests = append(m.capturedRequests, req)
	return m.response, m.err
}

func testClient() *Client {
	c := NewClient("testSpaceKey", "testApiKey")
	c.client = &mockHTTPClient{response: &http.Response{StatusCode: 200, Body: ioutil.NopCloser(bytes.NewBuffer(nil))}}
	return c
}
