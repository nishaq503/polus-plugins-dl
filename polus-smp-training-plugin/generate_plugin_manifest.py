import json
from typing import Any
from typing import List

from src import utils

INPUTS = {
    'pretrainedModel': {
        'description': (
            'Path to a model that was previously trained with this plugin. '
            'If starting fresh, you must instead provide: '
            '\'modelName\', '
            '\'encoderBase\', '
            '\'encoderVariant\', '
            '\'encoderWeights\', and '
            '\'optimizerName\'.'
            'See the README for available options.'
        ),
        'type': 'genericData',
        'required': False,
    },
    'modelName': {
        'description': 'Model architecture to use. Required if starting fresh.',
        'type': 'enum',
        'required': False,
        'options': {
            'values': utils.MODEL_NAMES,
        },
    },
    'encoderBase': {
        'description': 'Base encoder to use. Required if starting fresh.',
        'type': 'enum',
        'required': False,
        'options': {
            'values': utils.BASE_ENCODERS,
        },
    },
    'encoderVariant': {
        'description': 'Encoder Variant to use. Required if starting fresh.',
        'type': 'enum',
        'required': False,
        'options': {
            # 'values': utils.ENCODER_VARIANTS,
            'values': list(),
        },
    },
    'encoderWeights': {
        'description': (
            'Name of dataset with which the model was pretrained. '
            'Required if starting fresh.'
        ),
        'type': 'enum',
        'required': False,
        'options': {
            # 'values': utils.ENCODER_WEIGHTS,
            'values': list(),
        },
    },
    'optimizerName': {
        'description': (
            'Name of optimization algorithm to use for training the model. '
            'Required if starting fresh.'
        ),
        'type': 'enum',
        'required': False,
        'options': {
            'values': utils.OPTIMIZER_NAMES,
        },
    },

    'batchSize': {
        'description': (
            'Size of each batch for training. If left unspecified, we will automatically use '
            'the largest possible size based on the model architecture and GPU memory.'
        ),
        'type': 'number',
        'required': False,
    },

    'imagesDir': {
        'description': 'Collection containing images.',
        'type': 'collection',
        'required': True,
    },
    'imagesPattern': {
        'description': 'Filename pattern for images.',
        'type': 'string',
        'required': True,
    },
    'labelsDir': {
        'description': 'Collection containing labels, i.e. the ground-truth, for the images.',
        'type': 'collection',
        'required': True,
    },
    'labelsPattern': {
        'description': 'Filename pattern for labels.',
        'type': 'string',
        'required': True,
    },
    'trainFraction': {
        'description': 'Fraction of dataset to use for training.',
        'type': 'number',
        'required': True,
    },

    'lossName': {
        'description': 'Name of loss function to use.',
        'type': 'enum',
        'required': False,
        'options': {
            'values': utils.LOSS_NAMES,
        },
    },
    'metricName': {
        'description': 'Name of performance metric to track.',
        'type': 'enum',
        'required': False,
        'options': {
            'values': utils.METRIC_NAMES,
        },
    },
    'maxEpochs': {
        'description': 'Maximum number of epochs for which to continue training the model.',
        'type': 'number',
        'required': False,
    },
    'patience': {
        'description': 'Maximum number of epochs to wait for model to improve.',
        'type': 'number',
        'required': False,
    },
    'minDelta': {
        'description': 'Minimum improvement in loss to reset patience.',
        'type': 'number',
        'required': False,
    },
}

UI = list()


def bump_version(manifest):
    with open('VERSION', 'r') as infile:
        version = infile.read()

    [version, debug] = version.split('debug')
    version = f'{version}debug{str(1 + int(debug))}'

    with open('VERSION', 'w') as outfile:
        outfile.write(version)

    manifest['version'] = version
    manifest['containerId'] = f"{manifest['containerId'].split('0')[0]}:{version}"
    return manifest


def validate_variant() -> List[Any]:
    validator = list()
    for base, variants in utils.ENCODERS.items():
        condition = {
            'input': 'encoderBase',
            'value': base,
            'eval': '==',
        }
        then = list()
        for variant in variants.keys():
            then.append({
                'action': 'add',
                'input': 'encoderVariant',
                'value': variant,
            })

        validator.append({
            'condition': condition,
            'then': then,
        })

    return validator


def validate_weights() -> List[Any]:
    validator = list()

    for base, variants in utils.ENCODERS.items():
        for variant, weights in variants.items():
            condition = {
                'input': 'encoderWeights',
                'value': variant,
                'eval': '==',
            }
            then = list()
            for weight in weights:
                then.append({
                    'action': 'add',
                    'input': 'encoderWeights',
                    'value': weight,
                })

            validator.append({
                'condition': condition,
                'then': then,
            })

    return validator


def populate_ui():
    for key, values in INPUTS.items():
        field = {
            'key': f'inputs.{key}',
            'title': key,
            'description': values['description'],
        }
        if key == 'encoderVariant':
            field['validator'] = validate_variant()
        elif key == 'encoderWeights':
            field['validator'] = validate_weights()
        UI.append(field)

    return


def replace_manifest():
    with open('plugin.json', 'r') as infile:
        manifest = json.load(infile)

    manifest = bump_version(manifest)
    manifest['inputs'] = INPUTS
    manifest['ui'] = UI

    with open('new_plugin.json', 'w') as outfile:
        json.dump(manifest, outfile, indent=4)

    return


if __name__ == '__main__':
    populate_ui()
    replace_manifest()
