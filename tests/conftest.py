import pytest

from mdq import synthetic, store


@pytest.fixture
def panel():
    return synthetic.make_panel(n_securities=3, start="2022-01-03", end="2022-12-30", seed=3)


@pytest.fixture
def series():
    df = synthetic.make_series("SYN", start="2022-01-03", end="2022-12-30", seed=11)
    df["source_id"] = "a"
    df["adjustment"] = "split"
    return df


@pytest.fixture
def con():
    c = store.connect(":memory:")
    yield c
    c.close()
