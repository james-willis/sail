"""
Advanced Sedona SQL tests: spatial queries, Spark SQL combos, spatial joins.
"""
from pyspark.sql import SparkSession

spark = SparkSession.builder.remote("sc://localhost:50051").getOrCreate()

# ============================================================
# 1. SPATIAL QUERIES
# ============================================================
print("=" * 60)
print("1. SPATIAL QUERIES")
print("=" * 60)

print("\n--- Spatial filtering: points inside a polygon ---")
spark.sql("""
    WITH cities AS (
        SELECT 'New York' as name, ST_Point(-74.006, 40.7128) as geom
        UNION ALL SELECT 'Los Angeles', ST_Point(-118.2437, 33.9425)
        UNION ALL SELECT 'Chicago', ST_Point(-87.6298, 41.8781)
        UNION ALL SELECT 'London', ST_Point(-0.1278, 51.5074)
        UNION ALL SELECT 'Tokyo', ST_Point(139.6917, 35.6895)
    )
    SELECT name, ST_X(geom) as lon, ST_Y(geom) as lat
    FROM cities
    WHERE ST_Contains(
        ST_GeomFromWKT('POLYGON((-130 20, -60 20, -60 55, -130 55, -130 20))'),
        geom
    )
""").show(truncate=False)

print("--- Nearest neighbor: distance from origin ---")
spark.sql("""
    WITH places AS (
        SELECT 'A' as id, ST_Point(1.0, 1.0) as geom
        UNION ALL SELECT 'B', ST_Point(5.0, 5.0)
        UNION ALL SELECT 'C', ST_Point(2.0, 3.0)
        UNION ALL SELECT 'D', ST_Point(10.0, 10.0)
    )
    SELECT id,
           ROUND(ST_Distance(geom, ST_Point(0.0, 0.0)), 2) as dist_from_origin
    FROM places
    ORDER BY dist_from_origin
""").show(truncate=False)

print("--- Buffer + intersection area ---")
spark.sql("""
    SELECT ROUND(ST_Area(
        ST_Intersection(
            ST_Buffer(ST_Point(0.0, 0.0), 5.0),
            ST_Buffer(ST_Point(3.0, 0.0), 5.0)
        )
    ), 2) as overlap_area
""").show(truncate=False)

print("--- Geometry properties chain ---")
spark.sql("""
    WITH shapes AS (
        SELECT 'triangle' as name,
               ST_GeomFromWKT('POLYGON((0 0, 10 0, 5 8.66, 0 0))') as geom
        UNION ALL SELECT 'square',
               ST_GeomFromWKT('POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))')
        UNION ALL SELECT 'line',
               ST_GeomFromWKT('LINESTRING(0 0, 10 0, 10 10)')
    )
    SELECT name,
           ST_GeometryType(geom) as type,
           ROUND(ST_Area(geom), 2) as area,
           ROUND(ST_Length(geom), 2) as length,
           ST_IsValid(geom) as valid,
           ST_Dimension(geom) as dim
    FROM shapes
""").show(truncate=False)

# ============================================================
# 2. SPARK SQL + SPATIAL COMBOS
# ============================================================
print("=" * 60)
print("2. SPARK SQL + SPATIAL COMBOS")
print("=" * 60)

print("\n--- Array of geometries + spatial functions ---")
spark.sql("""
    SELECT
        array(ST_AsText(ST_Point(1.0, 2.0)),
              ST_AsText(ST_Point(3.0, 4.0)),
              ST_AsText(ST_Point(5.0, 6.0))) as point_texts,
        size(array(ST_Point(1.0, 2.0), ST_Point(3.0, 4.0), ST_Point(5.0, 6.0))) as num_points
""").show(truncate=False)

print("--- CASE WHEN with spatial predicates ---")
spark.sql("""
    WITH poi AS (
        SELECT 'Central Park' as name, ST_Point(-73.9654, 40.7829) as geom
        UNION ALL SELECT 'Times Square', ST_Point(-73.9855, 40.7580)
        UNION ALL SELECT 'Statue of Liberty', ST_Point(-74.0445, 40.6892)
    ),
    zones AS (
        SELECT ST_GeomFromWKT('POLYGON((-74.02 40.7, -73.95 40.7, -73.95 40.8, -74.02 40.8, -74.02 40.7))') as midtown
    )
    SELECT
        poi.name,
        CASE
            WHEN ST_Contains(zones.midtown, poi.geom) THEN 'Midtown'
            ELSE 'Other'
        END as zone
    FROM poi CROSS JOIN zones
""").show(truncate=False)

print("--- Aggregate: collect geometries + envelope ---")
spark.sql("""
    WITH pts AS (
        SELECT ST_Point(1.0, 1.0) as geom
        UNION ALL SELECT ST_Point(5.0, 5.0)
        UNION ALL SELECT ST_Point(3.0, 8.0)
        UNION ALL SELECT ST_Point(9.0, 2.0)
    )
    SELECT ST_AsText(ST_Envelope_Agg(geom)) as bounding_box
    FROM pts
""").show(truncate=False)

print("--- GROUP BY with spatial aggregates ---")
spark.sql("""
    WITH buildings AS (
        SELECT 'residential' as type, ST_Point(1.0, 1.0) as geom
        UNION ALL SELECT 'residential', ST_Point(2.0, 2.0)
        UNION ALL SELECT 'residential', ST_Point(3.0, 1.5)
        UNION ALL SELECT 'commercial', ST_Point(10.0, 10.0)
        UNION ALL SELECT 'commercial', ST_Point(11.0, 11.0)
    )
    SELECT
        type,
        count(*) as cnt,
        ST_AsText(ST_Envelope_Agg(geom)) as extent
    FROM buildings
    GROUP BY type
    ORDER BY type
""").show(truncate=False)

print("--- Window functions with spatial data ---")
spark.sql("""
    WITH sensors AS (
        SELECT 1 as id, 'zone_a' as zone, ST_Point(1.0, 1.0) as geom, 25.0 as temp
        UNION ALL SELECT 2, 'zone_a', ST_Point(2.0, 2.0), 27.0
        UNION ALL SELECT 3, 'zone_b', ST_Point(10.0, 10.0), 30.0
        UNION ALL SELECT 4, 'zone_b', ST_Point(11.0, 11.0), 32.0
    )
    SELECT
        id, zone, temp,
        ROUND(AVG(temp) OVER (PARTITION BY zone), 1) as zone_avg_temp,
        ROUND(ST_X(geom), 1) as x,
        ROUND(ST_Y(geom), 1) as y
    FROM sensors
    ORDER BY zone, id
""").show(truncate=False)

print("--- Explode + spatial: expand multi-geometries ---")
spark.sql("""
    SELECT ST_AsText(
        ST_GeomFromWKT(geom_wkt)
    ) as geom
    FROM (
        SELECT explode(array(
            'POINT(1 2)',
            'LINESTRING(0 0, 5 5)',
            'POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))'
        )) as geom_wkt
    )
""").show(truncate=False)

# ============================================================
# 3. SPATIAL JOIN
# ============================================================
print("=" * 60)
print("3. SPATIAL JOIN")
print("=" * 60)

print("\n--- Point-in-polygon spatial join ---")
# Create temp views for the join
spark.sql("""
    SELECT 'NYC' as city, ST_Point(-74.006, 40.7128) as geom
    UNION ALL SELECT 'LA', ST_Point(-118.2437, 33.9425)
    UNION ALL SELECT 'Chicago', ST_Point(-87.6298, 41.8781)
    UNION ALL SELECT 'Houston', ST_Point(-95.3698, 29.7604)
    UNION ALL SELECT 'Phoenix', ST_Point(-112.074, 33.4484)
""").createOrReplaceTempView("cities")

spark.sql("""
    SELECT 'Northeast' as region,
           ST_GeomFromWKT('POLYGON((-80 38, -66 38, -66 48, -80 48, -80 38))') as geom
    UNION ALL SELECT 'Midwest',
           ST_GeomFromWKT('POLYGON((-100 36, -80 36, -80 50, -100 50, -100 36))')
    UNION ALL SELECT 'West',
           ST_GeomFromWKT('POLYGON((-125 30, -100 30, -100 50, -125 50, -125 30))')
    UNION ALL SELECT 'South',
           ST_GeomFromWKT('POLYGON((-100 25, -80 25, -80 36, -100 36, -100 25))')
""").createOrReplaceTempView("regions")

spark.sql("""
    SELECT c.city, r.region
    FROM cities c
    JOIN regions r ON ST_Contains(r.geom, c.geom)
    ORDER BY c.city
""").show(truncate=False)

print("--- Distance-based spatial join (within 5 units) ---")
spark.sql("""
    WITH stores AS (
        SELECT 'Store A' as name, ST_Point(0.0, 0.0) as geom
        UNION ALL SELECT 'Store B', ST_Point(10.0, 10.0)
    ),
    customers AS (
        SELECT 'Alice' as name, ST_Point(1.0, 1.0) as geom
        UNION ALL SELECT 'Bob', ST_Point(3.0, 4.0)
        UNION ALL SELECT 'Carol', ST_Point(9.0, 9.0)
        UNION ALL SELECT 'Dave', ST_Point(11.0, 10.0)
    )
    SELECT
        c.name as customer,
        s.name as nearest_store,
        ROUND(ST_Distance(c.geom, s.geom), 2) as distance
    FROM customers c
    JOIN stores s ON ST_DWithin(c.geom, s.geom, 5.0)
    ORDER BY c.name, distance
""").show(truncate=False)

print("--- Polygon overlap spatial join ---")
spark.sql("""
    WITH parcels AS (
        SELECT 'Parcel 1' as name,
               ST_GeomFromWKT('POLYGON((0 0, 5 0, 5 5, 0 5, 0 0))') as geom
        UNION ALL SELECT 'Parcel 2',
               ST_GeomFromWKT('POLYGON((3 3, 8 3, 8 8, 3 8, 3 3))')
        UNION ALL SELECT 'Parcel 3',
               ST_GeomFromWKT('POLYGON((10 10, 15 10, 15 15, 10 15, 10 10))')
    ),
    flood_zone AS (
        SELECT ST_GeomFromWKT('POLYGON((2 2, 12 2, 12 12, 2 12, 2 2))') as geom
    )
    SELECT
        p.name,
        ROUND(ST_Area(ST_Intersection(p.geom, f.geom)), 2) as flooded_area,
        ROUND(ST_Area(p.geom), 2) as total_area,
        ROUND(ST_Area(ST_Intersection(p.geom, f.geom)) / ST_Area(p.geom) * 100, 1) as pct_flooded
    FROM parcels p
    JOIN flood_zone f ON ST_Intersects(p.geom, f.geom)
    ORDER BY pct_flooded DESC
""").show(truncate=False)

print("\n=== All advanced tests completed! ===")
spark.stop()
