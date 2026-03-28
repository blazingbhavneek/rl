from taskset import Problem, Score, BaseVerifier, BucketDistribution, CurriculumLoader, StatsWriter


def _run() -> None:
    p = Problem(id="x", statement="s", bucket=1, difficulty_label="easy")
    s = Score(compiled=True, passed=1, total=1)
    assert p.id == "x"
    assert s.total == 1
    _ = BaseVerifier
    _ = BucketDistribution
    _ = CurriculumLoader
    _ = StatsWriter


if __name__ == "__main__":
    _run()
    print("PASS: test_taskset_imports")
