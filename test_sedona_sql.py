"""
Test script for Sedona SQL functions via Sail's Spark Connect server.
Run after starting sail: sail spark server --port 50051
"""
from pyspark.sql import SparkSession

spark = SparkSession.builder.remote("sc://localhost:50051").getOrCreate()

print("=== Testing Sedona SQL functions via Sail ===\n")

# 1. Geometry constructors
print("--- ST_Point ---")
spark.sql("SELECT ST_AsText(ST_Point(1.0, 2.0)) as geom").show(truncate=False)

print("--- ST_GeomFromWKT ---")
spark.sql("SELECT ST_AsText(ST_GeomFromWKT('POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))')) as geom").show(truncate=False)

# 2. Spatial predicates
print("--- ST_Contains ---")
spark.sql("""
    SELECT ST_Contains(
        ST_GeomFromWKT('POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))'),
        ST_Point(5.0, 5.0)
    ) as contains_result
""").show(truncate=False)

# 3. Distance
print("--- ST_Distance ---")
spark.sql("SELECT ST_Distance(ST_Point(0.0, 0.0), ST_Point(3.0, 4.0)) as distance").show(truncate=False)

# 4. Buffer + Contains
print("--- ST_Buffer + ST_Contains ---")
spark.sql("""
    SELECT ST_Contains(
        ST_Buffer(ST_Point(0.0, 0.0), 10.0),
        ST_Point(5.0, 5.0)
    ) as buffered_contains
""").show(truncate=False)

# 5. Geometry properties
print("--- ST_Area ---")
spark.sql("SELECT ST_Area(ST_GeomFromWKT('POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))')) as area").show(truncate=False)

print("--- ST_Length ---")
spark.sql("SELECT ST_Length(ST_GeomFromWKT('LINESTRING(0 0, 3 4)')) as length").show(truncate=False)

# 6. Coordinate access
print("--- ST_X, ST_Y ---")
spark.sql("SELECT ST_X(ST_Point(1.5, 2.5)) as x, ST_Y(ST_Point(1.5, 2.5)) as y").show(truncate=False)

# 7. Spatial operations
print("--- ST_Intersection ---")
spark.sql("""
    SELECT ST_AsText(ST_Intersection(
        ST_GeomFromWKT('POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))'),
        ST_GeomFromWKT('POLYGON((5 5, 15 5, 15 15, 5 15, 5 5))')
    )) as intersection
""").show(truncate=False)

# 8. Validation
print("--- ST_IsValid ---")
spark.sql("SELECT ST_IsValid(ST_GeomFromWKT('POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))')) as is_valid").show(truncate=False)

print("\n=== All Sedona SQL tests passed! ===")
spark.stop()
