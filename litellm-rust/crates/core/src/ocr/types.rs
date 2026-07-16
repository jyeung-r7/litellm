use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OcrRequestData {
    pub data: Value,
    pub files: Option<Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OcrResponseData {
    pub pages: Vec<Value>,
    pub model: String,
    pub document_annotation: Option<Value>,
    pub usage_info: Option<Value>,
    pub object: String,
    pub content: Option<Value>,
    pub tables: Option<Value>,
    pub key_value_pairs: Option<Value>,
}

impl OcrResponseData {
    pub fn into_json(self) -> Value {
        let base = [
            ("pages".to_string(), Value::Array(self.pages)),
            ("model".to_string(), Value::String(self.model)),
            (
                "document_annotation".to_string(),
                self.document_annotation.unwrap_or(Value::Null),
            ),
            (
                "usage_info".to_string(),
                self.usage_info.unwrap_or(Value::Null),
            ),
            ("object".to_string(), Value::String(self.object)),
        ];
        let extras = [
            ("content", self.content),
            ("tables", self.tables),
            ("keyValuePairs", self.key_value_pairs),
        ]
        .into_iter()
        .filter_map(|(key, value)| value.map(|value| (key.to_string(), value)));
        Value::Object(base.into_iter().chain(extras).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn into_json_omits_absent_extras_but_keeps_base_null_fields() {
        let response = OcrResponseData {
            pages: vec![json!({"index": 0})],
            model: "prebuilt-read".to_string(),
            document_annotation: None,
            usage_info: None,
            object: "ocr".to_string(),
            content: None,
            tables: None,
            key_value_pairs: None,
        };

        let value = response.into_json();
        let object = value.as_object().expect("object");
        assert_eq!(object.get("document_annotation"), Some(&Value::Null));
        assert_eq!(object.get("usage_info"), Some(&Value::Null));
        assert!(!object.contains_key("content"));
        assert!(!object.contains_key("tables"));
        assert!(!object.contains_key("keyValuePairs"));
    }

    #[test]
    fn into_json_includes_present_extras() {
        let response = OcrResponseData {
            pages: vec![],
            model: "prebuilt-layout".to_string(),
            document_annotation: None,
            usage_info: None,
            object: "ocr".to_string(),
            content: Some(json!("full text")),
            tables: Some(json!([{"rowCount": 1}])),
            key_value_pairs: Some(json!([{"key": {"content": "k"}}])),
        };

        let value = response.into_json();
        assert_eq!(value["content"], "full text");
        assert_eq!(value["tables"][0]["rowCount"], 1);
        assert_eq!(value["keyValuePairs"][0]["key"]["content"], "k");
    }
}
