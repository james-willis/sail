import pyspark.sql.connect.functions as F  # noqa: N812
import pytest
from pyspark.errors.exceptions.connect import SparkConnectGrpcException

from pysail.testing.spark.utils.common import is_jvm_spark

pytestmark = pytest.mark.skipif(is_jvm_spark(), reason="Sail only")


@pytest.fixture(scope="module")
def fail():
    @F.udf("long")
    def _fail(_value):
        msg = "expected failure"
        raise RuntimeError(msg)

    return _fail


def test_execution_error_without_reattachment(spark_session_factory, fail, tmp_path):
    """Error handling for reattachable execution (the default) is exercised frequently in other tests.
    This test ensures that the error can still be propagated for (non-default) non-reattachable execution.
    """
    spark = spark_session_factory()
    spark._client.disable_reattachable_execute()  # noqa: SLF001
    # The UDF raises RuntimeError("expected failure"); the server truncates long
    # error messages, so match on the stable prefix of the worker traceback.
    match = "thrown from the Python worker"
    with pytest.raises(SparkConnectGrpcException, match=match):
        spark.range(1).select(fail("id")).write.mode("overwrite").parquet(str(tmp_path / "out"))
    with pytest.raises(SparkConnectGrpcException, match=match):
        spark.range(1).select(fail("id")).collect()


def test_write_command_is_reattachable(spark_session_factory, tmp_path):
    """Commands (e.g. writes) execute through the reattachable operation machinery,
    so a write must succeed and produce readable output under the default
    reattachable execution mode."""
    spark = spark_session_factory()
    path = str(tmp_path / "roundtrip")
    spark.range(10).write.mode("overwrite").parquet(path)
    assert spark.read.parquet(path).count() == 10
