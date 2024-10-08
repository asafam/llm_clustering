from enum import Enum
from clustering.constraints import *
from clustering.utils import *


class ConstraintsType(Enum):
    HardLabelsConstraints='HardLabelsConstraints',
    HardLabelsExcludeUncertainConstraints='HardLabelsExcludeUncertainConstraints',
    FuzzyLabelsConstraints='FuzzyLabelsConstraints',
    MustLinkCannotLinkConstraints='MustLinkCannotLinkConstraints',
    KConstraint='KConstraint',


class KInformationType(Enum):
    GroundTruthK='GroundTruthK',
    OracleK='OracleK'
    UnknownK='UnknownK'


def generate_constraint(data: any, constraint_type: ConstraintsType, **kwargs):
    if constraint_type in [ConstraintsType.HardLabelsConstraints, ConstraintsType.HardLabelsExcludeUncertainConstraints]:
        instances = data.get('result')
        return HardLabelsClusteringContraints(instances=instances)
    elif constraint_type == ConstraintsType.FuzzyLabelsConstraints:
        instances = data.get('result')
        return FuzzyLabelsClusteringContraints(instances=data)
    elif constraint_type == ConstraintsType.MustLinkCannotLinkConstraints:
        must_link = data.get('must_link', [])
        cannot_link = data.get('cannot_link', [])
        if not must_link and not cannot_link:
            raise ValueError("Must link and cannot link constraints are both empty")
        return MustLinkCannotLinkInstanceLevelClusteringConstraints(must_link=must_link, cannot_link=cannot_link)
    elif constraint_type == ConstraintsType.KConstraint:
        k = data.get('result')
        return KClusteringContraints(k=k)
    else:
        raise ValueError(f'No constraint matching the passed type {constraint_type}')

