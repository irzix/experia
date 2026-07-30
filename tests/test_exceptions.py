from uuid import uuid4

import pytest

import experia
import experia.core as core
from experia.core.exceptions import (
    ConfigurationError,
    EvaluationError,
    EvaluationFailure,
    ExperiaError,
    FailureDetail,
    LifecycleError,
    SanitizationError,
    StorageError,
    UnavailableFeatureError,
)


@pytest.mark.parametrize("error_type", [StorageError, ConfigurationError])
def test_existing_errors_preserve_positional_arguments(error_type):
    error = error_type("legacy message", "legacy detail")

    assert error.args == ("legacy message", "legacy detail")
    assert isinstance(error, ExperiaError)


def test_contextual_configuration_error_names_required_extra():
    error = ConfigurationError(
        feature="langchain",
        parameter="dependency",
        extra="experia[langchain]",
    )

    assert error.feature == "langchain"
    assert error.parameter == "dependency"
    assert error.extra == "experia[langchain]"


def test_lifecycle_error_exposes_only_operation_and_state_context():
    error = LifecycleError(state="draining", operation="submit")

    assert error.state == "draining"
    assert error.operation == "submit"
    assert "draining" not in str(error)


def test_evaluation_failure_is_caught_as_existing_evaluation_error():
    job_id = uuid4()
    experience_id = uuid4()
    detail = FailureDetail(
        job_id=job_id,
        operation="evaluation",
        experience_id=experience_id,
        error_type="TimeoutError",
    )
    error = EvaluationFailure(
        job_id=job_id,
        operation="evaluation",
        experience_id=experience_id,
        failures=(detail,),
    )

    assert isinstance(error, EvaluationError)
    assert error.job_id == job_id
    assert error.operation == "evaluation"
    assert error.experience_id == experience_id
    assert error.failures == (detail,)


def test_storage_error_normalizes_record_identifiers_without_payload_context():
    first_id = uuid4()
    error = StorageError(
        operation="decode",
        table="memories",
        record_ids=(first_id, "second-id"),
        migration="v1_to_v2",
        field="embedding",
    )

    assert error.operation == "decode"
    assert error.table == "memories"
    assert error.record_ids == (str(first_id), "second-id")
    assert error.migration == "v1_to_v2"
    assert error.field == "embedding"
    assert "second-id" not in str(error)


def test_storage_error_treats_one_string_identifier_as_one_record():
    error = StorageError(record_ids="record-1")

    assert error.record_ids == ("record-1",)


def test_sanitization_error_has_immutable_path_context():
    path = ["context", 0, "token"]
    error = SanitizationError(path=path, operation="external_request")
    path.append("later")

    assert error.path == ("context", 0, "token")
    assert error.operation == "external_request"
    assert "token" not in str(error)


def test_unavailable_feature_error_is_a_configuration_error():
    error = UnavailableFeatureError("crewai")

    assert isinstance(error, ConfigurationError)
    assert error.feature == "crewai"
    assert error.status == "planned"


@pytest.mark.parametrize(
    "name",
    [
        "ConfigurationError",
        "EvaluationError",
        "EvaluationFailure",
        "ExperiaError",
        "FailureDetail",
        "LifecycleError",
        "SanitizationError",
        "StorageError",
        "UnavailableFeatureError",
    ],
)
def test_error_types_are_exported_through_package_namespaces(name):
    expected = getattr(experia, name)

    assert getattr(core, name) is expected
    assert name in experia.__all__
    assert name in core.__all__
