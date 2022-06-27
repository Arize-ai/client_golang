<div align="center">
  <img src="https://storage.googleapis.com/arize-assets/arize-logo-white.jpg" width="600" /><br><br>
</div>

[![Slack](https://img.shields.io/badge/slack-@arize-yellow.svg?logo=slack)](https://join.slack.com/t/arize-ai/shared_invite/zt-g9c1j1xs-aQEwOAkU4T2x5K8cqI1Xqg)
[![license](https://img.shields.io/github/license/arize-ai/client_java)](https://github.com/Arize-ai/client_java/blob/main/LICENSE)
----
## Overview
A helper library to interact with Arize AI APIs.

Arize is an end-to-end ML observability and model monitoring platform. The platform is designed to help ML engineers and data science practitioners surface and fix issues with ML models in production faster with:
- Automated ML monitoring and model monitoring
- Workflows to troubleshoot model performance
- Real-time visualizations for model performance monitoring, data quality monitoring, and drift monitoring
- Model prediction cohort analysis
- Pre-deployment model validation
- Integrated model explainability

---
## Quickstart
This guide will help you instrument your code to log observability data for model monitoring and ML observability. The types of data supported include prediction labels, human readable/debuggable model features and tags, actual labels (once the ground truth is learned), and other model-related data. Logging model data allows you to generate powerful visualizations in the Arize platform to better monitor model performance, understand issues that arise, and debug your model's behavior. Additionally, Arize provides data quality monitoring, data drift detection, and performance management of your production models.

Start logging your model data with the following steps:

### 1. Sign up for your account
Sign up for a free account at https://arize.com/join.

<div align="center">
  <img src="https://storage.googleapis.com/arize-assets/Arize%20UI%20platform.jpg" /><br><br>
</div>

### 2. Get your service API key
When you create an account, we generate a service API key. You will need this API Key and your Space Key for logging authentication.

<div align="center">
  <img src="https://storage.googleapis.com/arize-assets/fixtures/copy-keys.png" /><br><br>
</div>

### Initialize Golang Client

Initialize `arize` at the start of your service using your previously created API Key and Space Key.

> **_NOTE:_** We strongly suggest storing the API key as a secret.

```golang
package main

import (
   "github.com/Arize-ai/client_golang"
)

func main() {
	c := arize.NewClient("YOUR_SPACE_KEY", "YOUR_API_KEY")
}
```

### Collect your model input features and labels you'd like to track

#### Real-time single prediction:
For a single real-time prediction, you can track all input features used at prediction time by logging them via a key:value map.

```golang

package main

import (
   "context"
   "fmt"
   "github.com/Arize-ai/client_golang"
   "github.com/google/uuid"
   "net/http"
   "time"
)

func main() {
   c := arize.NewClient("YOUR_SPACE_KEY", "YOUR_API_KEY")

   modelVersion := "v1"
   features := map[string]interface{}{"exampleFeatureName": 0.5}
   shapValues := map[string]float64{"exampleFeatureName": 1.0}
   eventMetadata := map[string]interface{}{"exampleEventMetadata": "xyz"}
   prediction := 0.9
   actual := 1.0
   now := time.Now()

   resp, err := c.Log(context.Background(), "exampleModelId", &modelVersion, uuid.NewString(), features, eventMetadata, shapValues, prediction, actual, &now, nil)
   if err != nil {
	   fmt.Printf("Log failed with err=%v \n", err)
   }
   if resp.StatusCode != http.StatusOK {
	   fmt.Printf("Request failed with status=%v, body=%v\n")
   }
   fmt.Println("Successfully logged a record to Arize")
}
```
