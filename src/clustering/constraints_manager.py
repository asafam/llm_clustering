from enum import Enum
from clustering.constraints import *
from clustering.utils import *


class ConstraintsType(Enum):
    HardLabelsConstraints='HardLabelsConstraints',
    HardLabelsExcludeUncertainConstraints='HardLabelsExcludeUncertainConstraints',
    FuzzyLabelsConstraints='FuzzyLabelsConstraints',
    MustLinkCannotLinkConstraints='MustLinkCannotLinkConstraints',
    KCountConstraint='KCountConstraint',
    KNameConstraint='KNameConstraint',


class KInformationType(Enum):
    GroundTruthK='GroundTruthK',
    OracleK='OracleK'
    UnknownK='UnknownK'


def generate_constraint(data: any, constraint_type: ConstraintsType, **kwargs) -> ClusteringConstraints:
    if constraint_type in [ConstraintsType.HardLabelsConstraints, ConstraintsType.HardLabelsExcludeUncertainConstraints]:
        sentences_labels = data.get('sentences_labels')
        return HardLabelsClusteringContraints(sentences_labels=sentences_labels)
    elif constraint_type == ConstraintsType.FuzzyLabelsConstraints:
        sentences_labels = data.get('sentences_labels')
        return FuzzyLabelsClusteringContraints(sentences_labels=sentences_labels)
    elif constraint_type == ConstraintsType.MustLinkCannotLinkConstraints:
        must_link = data.get('must_link', [])
        cannot_link = data.get('cannot_link', [])
        if not must_link and not cannot_link:
            raise ValueError("Must link and cannot link constraints are both empty")
        return MustLinkCannotLinkInstanceLevelClusteringConstraints(must_link=must_link, cannot_link=cannot_link)
    elif constraint_type == ConstraintsType.KCountConstraint:
        k = data.get('k')
        return KClusteringContraints(k=k)
    elif constraint_type == ConstraintsType.KNameConstraint:
        label_names = data.get('result')
        return KClusteringContraints(label_names=label_names)
    else:
        raise ValueError(f'No constraint matching the passed type {constraint_type}')

