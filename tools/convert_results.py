"""Format converter for the predicted bbox json file.

Convert the predicted bbox json file from the coco format to the required one
(let's call it xyxy format) for the evaluation. There are three differences:

1. coco: [xmin, ymin, width, height] -> xyxy: [xmin, ymin, xmax, ymax]
2. coco: list of dict -> xyxy: dict of dict of list
3. coco: image_id -> xyxy: file_name

Please check `sample_submission.json`, which is in the xyxy format.

Usage:
    python convert_results.py <coco_json_path> <annot_json_path>
"""

import json
import sys


def main():
    coco_json_path: str = sys.argv[1]
    annot_json_path: str = sys.argv[2]
    with open(coco_json_path, 'r') as f:
        coco_json = json.load(f)

    map_id_to_filename = _create_id_to_filename_dict(annot_json_path)

    output_json = {}
    for item in coco_json:
        file_name = map_id_to_filename[item['image_id']]
        if file_name not in output_json:
            output_json[file_name] = {
                'boxes': [],
                'labels': [],
                'scores': [],
            }
        output_json[file_name]['boxes'].append([
            item['bbox'][0],
            item['bbox'][1],
            item['bbox'][0] + item['bbox'][2],
            item['bbox'][1] + item['bbox'][3],
        ])
        output_json[file_name]['labels'].append(item['category_id'])
        output_json[file_name]['scores'].append(item['score'])
    assert set(output_json.keys()) == set(map_id_to_filename.values())

    with open(coco_json_path.replace('.json', '.converted.json'), 'w') as f:
        json.dump(output_json, f, indent=4)


def _create_id_to_filename_dict(annotation_path):
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    id_to_filename = {}
    for image_info in annotation['images']:
        id_to_filename[image_info['id']] = image_info['file_name']
    return id_to_filename


if __name__ == '__main__':
    main()
