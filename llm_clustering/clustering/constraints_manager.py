from enum import Enum
from constraints import *


class ConstraintsType(Enum):
    HardLabelsConstraints='HardLabelsConstraints',
    FuzzyLabelsConstraints='FuzzyLabelsConstraints',
    MustLinkCannotLinkConstraints='MustLinkCannotLinkConstraints',


class KInformationType(Enum):
    GroundTruthK='GroundTruthK',
    OracleK='OracleK'
    UnknownK='UnknownK'


def generate_constraint(data: any, constraint_type: ConstraintsType):
    if constraint_type == ConstraintsType.HardLabelsConstraints:
        return HardLabelsClusteringContraints(instances=data)
    elif constraint_type == ConstraintsType.FuzzyLabelsConstraints:
        return FuzzyLabelsClusteringContraints(instances=data)
    elif constraint_type == ConstraintsType.MustLinkCannotLinkConstraints:
        must_link = data.get('must_link', [])
        cannot_link = data.get('cannot_link', [])
        if not must_link and not cannot_link:
            raise ValueError("Must link and cannot link constraints are both empty")
        return MustLinkCannotLinkInstanceLevelClusteringConstraints(must_link=must_link, cannot_link=cannot_link)

