from evalset.samples import devops, lintao, mia, nostamp, xiaoyu
from evalset.schema import Sample

ALL: tuple[Sample, ...] = (mia.SAMPLE, lintao.SAMPLE, devops.SAMPLE, nostamp.SAMPLE, xiaoyu.SAMPLE)


def by_name(names: list[str] | None = None) -> list[Sample]:
    if not names:
        return list(ALL)
    index = {s.name: s for s in ALL}
    missing = [n for n in names if n not in index]
    if missing:
        raise KeyError(f"unknown samples {missing}; have {sorted(index)}")
    return [index[n] for n in names]
