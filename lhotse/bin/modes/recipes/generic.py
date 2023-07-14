import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.generic import download_generic, prepare_generic
from lhotse.utils import Pathlike


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
def generic(target_dir: Pathlike):
    """generic dataset download."""
    download_generic(target_dir)


@prepare.command(context_settings=dict(show_default=True))
@click.option("-cj", "--corpus_jsonl", type=str)
@click.option("-od", "--output_dir", type=str, default='data')
@click.option("-c", "--corpus_name", type=str)
@click.option("-w", "--wer_threshold", type=float, default=100.0)
@click.option("-s", "--sample_rate", type=int, default=16000)
@click.option("-n", "--num_procs", type=int, default=1)
def generic(corpus_jsonl: str, corpus_name: str,
            output_dir: Pathlike, wer_threshold: float = 100.0,
            sample_rate: int = 16000, num_procs: int = 1
):
    """generic data preparation."""
    prepare_generic(
        corpus_jsonl,
        corpus_name,
        output_dir=output_dir,
        wer_threshold=wer_threshold,
        sample_rate=sample_rate,
        num_procs=num_procs
    )
