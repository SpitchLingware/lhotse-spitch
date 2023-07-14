"""
About the generic dataset:

This dataset was created to emulate the yes-no project and support 
a generic dataset similar to the audio-folder dataset approach from Huggingface:
  - https://huggingface.co/docs/datasets/audio_dataset#audiofolder

This approach expects a simple folder-based setup where the user has 
provided all information necessary in the folder.  The purpose of this approach
is to simplify common datasets and provide an entrypoint that does not 
require modifying the lhotse module in order to add a recipe.

As a consequence of the above the generic dataset expects all necessary
configuration to be pre-organized in the target audiofolder, just like the
approach provided by the HuggingFace AudioFolder.
"""

import logging
import shutil
import tarfile
import os
import json
import re
from copy import deepcopy
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, safe_extract

_DEFAULT_URL = ""


def download_generic(
    audio_tar: str,
    target_dir: Pathlike = ".",
    force_download: Optional[bool] = False,
    url: Optional[str] = _DEFAULT_URL,
) -> Path:
    """Download and untar the dataset.
    :param audio_tar: str, the name of an audio tarball.
    :param target_dir: Pathlike, the path of the dir to store the dataset.
        The extracted files are saved to target_dir/audio/*.flac
    :param force_download: Bool, if True, download the tar file no matter
        whether it exists or not.
    :param url: str, the url to download the dataset.
    :return: the path to downloaded and extracted directory with data.
    """
    logging.info(f"Skipping - content exists.")

    return ''


def _prepare_datum(
    datum: Dict[str, Any]
) -> Tuple[Recording, SupervisionSegment]:
    """Build a list of Recording and SupervisionSegment from a list
    of sound filenames.
  
    :param datum: Dict[str, Any], a list of sound filenames
    :return: a tuple containing a list of Recording and a list
        of SupervisionSegment
    """
    recording = Recording.from_file(
        datum['location'],
        recording_id=datum['fid']
    )

    segment = SupervisionSegment(
        id=datum['fid'],
        recording_id=datum['fid'],
        start=0.0,
        duration=recording.duration,
        channel=0,
        language=datum['lang'],
        text=datum['clean_ref'],
    )

    return recording, segment

chars = defaultdict(int)


def prepare_generic(
        corpus_jsonl: str, corpus_name: str,
        output_dir: Optional[Pathlike] = None,
        wer_threshold: float = 100.0,
        sample_rate: int = 16000,
        num_procs: int = 1
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply
    read and return them.

    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is either "train" or "test", and the value is
        Dicts with the keys 'recordings' and 'supervisions'.
    """
    #corpus_dir = Path(corpus_dir)
    #assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    #corpus_jsonl = os.path.join(corpus_dir, "corpus.jsonl")
    assert os.path.exists(corpus_jsonl)
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    manifests = defaultdict(dict)

    meta_corpus = defaultdict(lambda: defaultdict(list))
    with open(corpus_jsonl) as ifp:
        for idx,record in enumerate(ifp):
            record = json.loads(record.strip())
            if record.get('wer', 100.0) > wer_threshold:
                continue

            if re.match(r"^\s*$", record.get('clean_ref', '')):
                continue
            if record.get('path_duration', 0.0) < 1.0 or record.get('path_duration', 0.0) > 20.0:
                continue
            #if record.get('location', '') == '':
            #    continue
            for lkey in ['orig_path', 'segment_path', 'opus_segment_path']:
                if lkey in record:
                    new_record = deepcopy(record)
                    if new_record[lkey] == True:
                        new_record['location'] = record['location']
                    else:
                        new_record['location'] = new_record[lkey]
                    new_record['fid'] = f"{record['fid']}_{lkey}"
                    if True:
                        recording, segment = _prepare_datum(new_record)
                        meta_corpus[record['split']]['recordings'].append(recording)
                        meta_corpus[record['split']]['segments'].append(segment)
                    else:
                        print("Skipping:", json.dumps(new_record, ensure_ascii=False))

    for ch,cnt in sorted(chars.items(), key=lambda x: x[1]):
        print(f"{ch}: {cnt}")
        
        
    for name, dataset in meta_corpus.items():
        recording_set = RecordingSet.from_recordings(
            dataset['recordings']
        )
        recording_set = recording_set.resample(sample_rate)

        supervision_set = SupervisionSet.from_segments(
            dataset['segments']
        )

        validate_recordings_and_supervisions(
            recording_set,
            supervision_set
        )

        recording_set.to_file(
            os.path.join(
                output_dir,
                f"{corpus_name}_recordings_{name}.jsonl.gz"
            )
        )

        supervision_set.to_file(
            os.path.join(
                output_dir,
                f"{corpus_name}_supervisions_{name}.jsonl.gz"
            )
        )

        manifests[name] = {
            "recordings": recording_set,
            "supervisions": supervision_set
        }

    return manifests


if __name__ == "__main__":
    import sys
    import argparse

    example = f"{sys.argv[0]} --corpus_dir dir --corpus_name corpus"
    parser = argparse.ArgumentParser(description=example)
    parser.add_argument("--corpus_dir", "-c", help="Corpus directory.",
                        required=True)
    parser.add_argument("--output_dir", "-od", help="Output directory.",
                        required=True)
    parser.add_argument("--corpus_name", "-n", help="Corpus name.",
                        default='corpus')
    parser.add_argument("--wer_threshold", "-w", help="WER threshold for inclusion.",
                        default=100.0, type=float)
    parser.add_argument("--sample_rate", "-s", help="Target sample rate.",
                        default=16000, type=int)
    args = parser.parse_args()
    
    prepare_generic(
        args.corpus_dir,
        args.corpus_name,
        args.output_dir,
        wer_threshold=args.wer_threshold,
        sample_rate=args.sample_rate
    )
