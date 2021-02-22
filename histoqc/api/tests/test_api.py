import types

import numpy as np
import pytest

import histoqc.api as qc


@pytest.fixture(scope="function")
def pipeline_state(svs_small, tmp_path):
    yield qc.PipelineState(fname=svs_small, fname_outdir=tmp_path, params={})


def test_pipeline_state(svs_small, tmp_path):
    s = qc.PipelineState(fname=svs_small, fname_outdir=tmp_path, params={})
    assert isinstance(s.mask, np.ndarray)
    assert np.all(s.mask), "initial mask is restricted?"


FUNCTIONS = [
    obj
    for obj in (getattr(qc, obj_name) for obj_name in qc.__all__)
    if isinstance(obj, types.FunctionType)
]


@pytest.mark.parametrize(
    "histoqc_func",
    argvalues=FUNCTIONS,
    ids=[func.__qualname__ for func in FUNCTIONS],
)
def test_calling_with_defaults(pipeline_state, histoqc_func):
    try:
        mask = histoqc_func(pipeline_state)
    except TypeError as err:
        if "required keyword-only" in str(err):
            pytest.skip("function has required default arguments")
        raise
    assert mask is not None


def test_pipeline_chain(pipeline_state):
    c = qc.PipelineChain()
    qc.get_contrast(c)
    qc.get_histogram(c)
    assert c.run(pipeline_state) is not None
