use std::collections::HashMap;

use datafusion::arrow::datatypes as adt;

use crate::error::{SparkError, SparkResult};
use crate::spark::connect::{data_type as sdt, DataType};

/// GeoArrow extension metadata extracted from Arrow field metadata.
#[derive(Debug, Clone)]
struct GeoArrowMetadata {
    /// Edge interpolation: "planar" for Geometry, "spherical" for Geography
    edges: String,
    /// Spatial Reference System Identifier (SRID). -1 for mixed SRID.
    srid: i32,
}

impl GeoArrowMetadata {
    /// Parse GeoArrow metadata from JSON string.
    /// Defaults to planar edges and SRID 4326 if parsing fails.
    ///
    /// Supports Spark 4.1 CRS formats:
    /// - "OGC:CRS84" → SRID 4326 (WGS84)
    /// - "EPSG:{srid}" → direct SRID mapping
    /// - "SRID:0" → SRID 0 (unspecified)
    /// - "SRID:ANY" → SRID -1 (mixed)
    fn from_json(metadata: &str) -> Self {
        let parsed = serde_json::from_str::<serde_json::Value>(metadata).ok();

        let edges = parsed
            .as_ref()
            .and_then(|v| v.get("edges").and_then(|e| e.as_str()))
            .unwrap_or("planar")
            .to_string();

        let srid = parsed
            .and_then(|v| {
                v.get("crs").and_then(|crs| crs.as_str()).and_then(|s| {
                    // Handle Spark 4.1 CRS string formats
                    match s {
                        "OGC:CRS84" => Some(4326),
                        "SRID:ANY" => Some(-1),
                        "SRID:0" => Some(0),
                        _ if s.starts_with("EPSG:") => {
                            s.strip_prefix("EPSG:").and_then(|n| n.parse::<i32>().ok())
                        }
                        _ if s.starts_with("SRID:") => {
                            s.strip_prefix("SRID:").and_then(|n| n.parse::<i32>().ok())
                        }
                        _ => None,
                    }
                })
            })
            .unwrap_or(4326);

        Self { edges, srid }
    }

    /// Returns true if this represents a Geography type (spherical edges).
    fn is_geography(&self) -> bool {
        self.edges == "spherical"
    }
}

impl TryFrom<adt::Field> for sdt::StructField {
    type Error = SparkError;

    fn try_from(field: adt::Field) -> SparkResult<sdt::StructField> {
        let is_udt = field.metadata().keys().any(|k| k.starts_with("udt."));
        let is_geoarrow = field
            .metadata()
            .get("ARROW:extension:name")
            .map(|v| v == "geoarrow.wkb")
            .unwrap_or(false);

        let data_type = if is_udt {
            DataType {
                kind: Some(sdt::Kind::Udt(Box::new(sdt::Udt {
                    r#type: "udt".to_string(),
                    jvm_class: field.metadata().get("udt.jvm_class").cloned(),
                    python_class: field.metadata().get("udt.python_class").cloned(),
                    serialized_python_class: field
                        .metadata()
                        .get("udt.serialized_python_class")
                        .cloned(),
                    sql_type: Some(Box::new(field.data_type().clone().try_into()?)),
                }))),
            }
        } else if is_geoarrow {
            // Parse geoarrow extension metadata to determine Geometry vs Geography
            let ext_metadata = field
                .metadata()
                .get("ARROW:extension:metadata")
                .cloned()
                .unwrap_or_default();
            let geo_meta = GeoArrowMetadata::from_json(&ext_metadata);

            if geo_meta.is_geography() {
                DataType {
                    kind: Some(sdt::Kind::Geography(sdt::Geography {
                        srid: geo_meta.srid,
                        type_variation_reference: 0,
                    })),
                }
            } else {
                DataType {
                    kind: Some(sdt::Kind::Geometry(sdt::Geometry {
                        srid: geo_meta.srid,
                        type_variation_reference: 0,
                    })),
                }
            }
        } else {
            field.data_type().clone().try_into()?
        };
        // FIXME: The metadata. prefix is managed by Sail and the convention should be respected everywhere.
        // Also filter out Arrow extension metadata (geoarrow, etc.) which is internal to Sail
        let metadata = &field
            .metadata()
            .iter()
            .filter(|(k, _)| !k.starts_with("udt.") && !k.starts_with("ARROW:extension:"))
            .map(|(k, v)| {
                let parsed = serde_json::from_str::<serde_json::Value>(v)
                    .unwrap_or_else(|_| serde_json::Value::String(v.clone()));
                Ok((k.strip_prefix("metadata.").unwrap_or(k), parsed))
            })
            .collect::<SparkResult<HashMap<_, serde_json::Value>>>()?;
        let metadata = serde_json::to_string(metadata)?;
        Ok(sdt::StructField {
            name: field.name().clone(),
            data_type: Some(data_type),
            nullable: field.is_nullable(),
            metadata: Some(metadata),
        })
    }
}

/// Reference: https://github.com/apache/spark/blob/bb17665955ad536d8c81605da9a59fb94b6e0162/sql/api/src/main/scala/org/apache/spark/sql/util/ArrowUtils.scala
impl TryFrom<adt::DataType> for DataType {
    type Error = SparkError;

    fn try_from(data_type: adt::DataType) -> SparkResult<DataType> {
        use sdt::Kind;

        let error =
            |x: &adt::DataType| SparkError::unsupported(format!("cast {x:?} to Spark data type"));
        let kind = match data_type {
            adt::DataType::Null => Kind::Null(sdt::Null::default()),
            adt::DataType::Binary
            | adt::DataType::FixedSizeBinary(_)
            | adt::DataType::LargeBinary
            | adt::DataType::BinaryView => Kind::Binary(sdt::Binary::default()),
            adt::DataType::Boolean => Kind::Boolean(sdt::Boolean::default()),
            // TODO: cast unsigned integer types to signed integer types in the query output,
            //   and return an error if unsigned integer types are found here.
            adt::DataType::UInt8 | adt::DataType::Int8 => Kind::Byte(sdt::Byte::default()),
            adt::DataType::UInt16 | adt::DataType::Int16 => Kind::Short(sdt::Short::default()),
            adt::DataType::UInt32 | adt::DataType::Int32 => Kind::Integer(sdt::Integer::default()),
            adt::DataType::UInt64 | adt::DataType::Int64 => Kind::Long(sdt::Long::default()),
            adt::DataType::Float16 => return Err(error(&data_type)),
            adt::DataType::Float32 => Kind::Float(sdt::Float::default()),
            adt::DataType::Float64 => Kind::Double(sdt::Double::default()),
            adt::DataType::Decimal128(precision, scale)
            | adt::DataType::Decimal256(precision, scale) => Kind::Decimal(sdt::Decimal {
                scale: Some(scale as i32),
                precision: Some(precision as i32),
                type_variation_reference: 0,
            }),
            // FIXME: This mapping might not always be correct due to converting to Arrow data types and back.
            //  For example, this originally may have been a `Kind::Char` or `Kind::VarChar` in Spark.
            //  We retain the original type information in the spec, but it is lost after converting to Arrow.
            adt::DataType::Utf8 | adt::DataType::LargeUtf8 | adt::DataType::Utf8View => {
                Kind::String(sdt::String::default())
            }
            adt::DataType::Date32 => Kind::Date(sdt::Date::default()),
            adt::DataType::Date64 | adt::DataType::Time32 { .. } | adt::DataType::Time64 { .. } => {
                return Err(error(&data_type))
            }
            adt::DataType::Timestamp(adt::TimeUnit::Microsecond, None) => {
                Kind::TimestampNtz(sdt::TimestampNtz::default())
            }
            adt::DataType::Timestamp(adt::TimeUnit::Microsecond, Some(_)) => {
                Kind::Timestamp(sdt::Timestamp::default())
            }
            adt::DataType::Timestamp(adt::TimeUnit::Second, _)
            | adt::DataType::Timestamp(adt::TimeUnit::Millisecond, _)
            | adt::DataType::Timestamp(adt::TimeUnit::Nanosecond, _) => {
                return Err(error(&data_type))
            }
            adt::DataType::Interval(adt::IntervalUnit::MonthDayNano) => {
                Kind::CalendarInterval(sdt::CalendarInterval::default())
            }
            adt::DataType::Interval(adt::IntervalUnit::YearMonth) => {
                Kind::YearMonthInterval(sdt::YearMonthInterval {
                    start_field: None,
                    end_field: None,
                    type_variation_reference: 0,
                })
            }
            adt::DataType::Interval(adt::IntervalUnit::DayTime) => {
                Kind::DayTimeInterval(sdt::DayTimeInterval {
                    start_field: None,
                    end_field: None,
                    type_variation_reference: 0,
                })
            }
            adt::DataType::Duration(adt::TimeUnit::Microsecond) => {
                Kind::DayTimeInterval(sdt::DayTimeInterval {
                    start_field: None,
                    end_field: None,
                    type_variation_reference: 0,
                })
            }
            adt::DataType::Duration(
                adt::TimeUnit::Second | adt::TimeUnit::Millisecond | adt::TimeUnit::Nanosecond,
            ) => return Err(error(&data_type)),
            adt::DataType::List(field)
            | adt::DataType::FixedSizeList(field, _)
            | adt::DataType::LargeList(field)
            | adt::DataType::ListView(field)
            | adt::DataType::LargeListView(field) => {
                let field = sdt::StructField::try_from(field.as_ref().clone())?;
                Kind::Array(Box::new(sdt::Array {
                    element_type: field.data_type.map(Box::new),
                    contains_null: field.nullable,
                    type_variation_reference: 0,
                }))
            }
            adt::DataType::Struct(fields) => Kind::Struct(sdt::Struct {
                fields: fields
                    .into_iter()
                    .map(|f| f.as_ref().clone().try_into())
                    .collect::<SparkResult<Vec<sdt::StructField>>>()?,
                type_variation_reference: 0,
            }),
            adt::DataType::Map(ref field, ref _keys_sorted) => {
                let field = sdt::StructField::try_from(field.as_ref().clone())?;
                let Some(DataType {
                    kind: Some(Kind::Struct(sdt::Struct { fields, .. })),
                }) = field.data_type
                else {
                    return Err(error(&data_type));
                };
                let [key_field, value_field] = fields.as_slice() else {
                    return Err(error(&data_type));
                };
                Kind::Map(Box::new(sdt::Map {
                    key_type: key_field.data_type.clone().map(Box::new),
                    value_type: value_field.data_type.clone().map(Box::new),
                    value_contains_null: value_field.nullable,
                    type_variation_reference: 0,
                }))
            }
            adt::DataType::Union { .. }
            | adt::DataType::Dictionary { .. }
            | adt::DataType::RunEndEncoded(_, _)
            | adt::DataType::Decimal32(_, _)
            | adt::DataType::Decimal64(_, _) => return Err(error(&data_type)),
        };
        Ok(DataType { kind: Some(kind) })
    }
}

#[cfg(test)]
mod tests {
    // Tests are allowed to use panic for assertions
    #![allow(clippy::panic)]

    use super::*;
    use crate::error::SparkResult;

    #[test]
    fn test_geoarrow_metadata_parsing() {
        // Test planar (Geometry default) with Spark 4.1 CRS format
        let metadata = r#"{"crs":"OGC:CRS84","edges":"planar"}"#;
        let parsed = GeoArrowMetadata::from_json(metadata);
        assert_eq!(parsed.edges, "planar");
        assert_eq!(parsed.srid, 4326);
        assert!(!parsed.is_geography());

        // Test spherical (Geography) with Spark 4.1 CRS format
        let metadata = r#"{"crs":"OGC:CRS84","edges":"spherical"}"#;
        let parsed = GeoArrowMetadata::from_json(metadata);
        assert_eq!(parsed.edges, "spherical");
        assert_eq!(parsed.srid, 4326);
        assert!(parsed.is_geography());

        // Test Web Mercator projection (Cartesian, valid for Geometry only)
        let metadata = r#"{"crs":"EPSG:3857","edges":"planar"}"#;
        let parsed = GeoArrowMetadata::from_json(metadata);
        assert_eq!(parsed.srid, 3857);

        // Test mixed SRID format
        let metadata = r#"{"crs":"SRID:ANY","edges":"planar"}"#;
        let parsed = GeoArrowMetadata::from_json(metadata);
        assert_eq!(parsed.srid, -1);

        // Test default when edges not present
        let metadata = r#"{"crs":"OGC:CRS84"}"#;
        let parsed = GeoArrowMetadata::from_json(metadata);
        assert_eq!(parsed.edges, "planar");

        // Test default when CRS not present
        let metadata = r#"{"edges":"planar"}"#;
        let parsed = GeoArrowMetadata::from_json(metadata);
        assert_eq!(parsed.srid, 4326);

        // Test default on invalid JSON
        let parsed = GeoArrowMetadata::from_json("invalid");
        assert_eq!(parsed.edges, "planar");
        assert_eq!(parsed.srid, 4326);
    }

    #[test]
    fn test_geoarrow_field_to_proto_geometry() -> SparkResult<()> {
        use crate::spark::connect::data_type::{Geometry, Kind};

        // Create an Arrow field with geoarrow.wkb metadata for Geometry
        // Note: CRS "OGC:CRS84" is the Spark 4.1 standard format for SRID 4326
        let metadata: HashMap<String, String> = [
            (
                "ARROW:extension:name".to_string(),
                "geoarrow.wkb".to_string(),
            ),
            (
                "ARROW:extension:metadata".to_string(),
                r#"{"crs":"OGC:CRS84","edges":"planar"}"#.to_string(),
            ),
        ]
        .into_iter()
        .collect();
        let field = adt::Field::new("geom", adt::DataType::Binary, true).with_metadata(metadata);

        let proto_field: sdt::StructField = field.try_into()?;

        assert_eq!(proto_field.name, "geom");
        match proto_field.data_type {
            Some(DataType {
                kind: Some(Kind::Geometry(Geometry { srid, .. })),
            }) => {
                assert_eq!(srid, 4326);
            }
            _ => panic!("Expected Geometry proto type"),
        }

        Ok(())
    }

    #[test]
    fn test_geoarrow_field_to_proto_geography() -> SparkResult<()> {
        use crate::spark::connect::data_type::{Geography, Kind};

        // Create an Arrow field with geoarrow.wkb metadata for Geography (spherical)
        // Note: In Spark 4.1, only SRID 4326 is valid for Geography.
        // SRID 3857 (Web Mercator) is valid for Geometry but NOT for Geography,
        // as it's a Cartesian projection, not a geographic CRS.
        // CRS "OGC:CRS84" is the Spark 4.1 standard format for SRID 4326
        let metadata: HashMap<String, String> = [
            (
                "ARROW:extension:name".to_string(),
                "geoarrow.wkb".to_string(),
            ),
            (
                "ARROW:extension:metadata".to_string(),
                r#"{"crs":"OGC:CRS84","edges":"spherical"}"#.to_string(),
            ),
        ]
        .into_iter()
        .collect();
        let field = adt::Field::new("geog", adt::DataType::Binary, true).with_metadata(metadata);

        let proto_field: sdt::StructField = field.try_into()?;

        assert_eq!(proto_field.name, "geog");
        match proto_field.data_type {
            Some(DataType {
                kind: Some(Kind::Geography(Geography { srid, .. })),
            }) => {
                assert_eq!(srid, 4326);
            }
            _ => panic!("Expected Geography proto type"),
        }

        Ok(())
    }

    #[test]
    fn test_geoarrow_field_mixed_srid() -> SparkResult<()> {
        use crate::spark::connect::data_type::{Geometry, Kind};

        // Test mixed SRID (-1) support
        let metadata: HashMap<String, String> = [
            (
                "ARROW:extension:name".to_string(),
                "geoarrow.wkb".to_string(),
            ),
            (
                "ARROW:extension:metadata".to_string(),
                r#"{"crs":"SRID:ANY","edges":"planar"}"#.to_string(),
            ),
        ]
        .into_iter()
        .collect();
        let field = adt::Field::new("geom", adt::DataType::Binary, true).with_metadata(metadata);

        let proto_field: sdt::StructField = field.try_into()?;

        match proto_field.data_type {
            Some(DataType {
                kind: Some(Kind::Geometry(Geometry { srid, .. })),
            }) => {
                assert_eq!(srid, -1);
            }
            _ => panic!("Expected Geometry proto type with mixed SRID"),
        }

        Ok(())
    }
}
