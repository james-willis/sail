"""
Verifies that spatial joins use Sedona's SpatialJoinExec rather than NestedLoopJoinExec.

Run after starting sail:
    sail spark server --port 50051

Then:
    python test_spatial_join_plan.py
"""
from pyspark.sql import SparkSession

spark = SparkSession.builder.remote("sc://localhost:50051").getOrCreate()

spark.sql("""
    SELECT 'NYC' as city, ST_Point(-74.006, 40.7128) as geom
    UNION ALL SELECT 'LA', ST_Point(-118.2437, 33.9425)
    UNION ALL SELECT 'Chicago', ST_Point(-87.6298, 41.8781)
""").createOrReplaceTempView("cities")

spark.sql("""
    SELECT 'Northeast' as region,
           ST_GeomFromWKT('POLYGON((-80 38, -66 38, -66 48, -80 48, -80 38))') as geom
    UNION ALL SELECT 'West',
           ST_GeomFromWKT('POLYGON((-125 30, -100 30, -100 50, -125 50, -125 30))')
""").createOrReplaceTempView("regions")

queries = {
    "ST_Intersects": """
        EXPLAIN SELECT c.city, r.region
        FROM cities c JOIN regions r ON ST_Intersects(c.geom, r.geom)
    """,
    "ST_Contains": """
        EXPLAIN SELECT c.city, r.region
        FROM cities c JOIN regions r ON ST_Contains(r.geom, c.geom)
    """,
    "ST_DWithin": """
        EXPLAIN SELECT a.city, b.city
        FROM cities a JOIN cities b ON ST_DWithin(a.geom, b.geom, 5.0)
    """,
}

failures = []
for label, sql in queries.items():
    print(f"\n=== {label} ===")
    rows = spark.sql(sql).collect()
    plan = "\n".join(r[0] for r in rows)
    print(plan)
    if "SpatialJoinExec" in plan:
        print(f"PASS: {label} uses SpatialJoinExec")
    elif "NestedLoopJoinExec" in plan:
        failures.append(f"{label}: still planning as NestedLoopJoinExec")
        print(f"FAIL: {label} still using NestedLoopJoinExec")
    else:
        failures.append(f"{label}: unknown join in plan (no SpatialJoinExec, no NestedLoopJoinExec)")
        print(f"UNKNOWN: {label} — neither SpatialJoinExec nor NestedLoopJoinExec found")

print("\n" + "=" * 60)
if failures:
    print("FAILURES:")
    for f in failures:
        print(f"  - {f}")
    raise SystemExit(1)
print("All spatial joins planned via SpatialJoinExec.")
