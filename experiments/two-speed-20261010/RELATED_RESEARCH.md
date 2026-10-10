# Primary-source context

Checked 2026-10-10 before evaluation:

- [Losing, Hammer and Wersing, SAM-kNN, IJCAI 2017](https://www.ijcai.org/Proceedings/2017/0690.pdf)
  uses short/long memories and past-performance selection for heterogeneous
  drift. It also cleans and compresses long-term knowledge. Our simple FIFO
  views and EMA selectors do not implement those algorithms or establish novelty.
- [River ADWIN documentation](https://riverml.xyz/latest/api/drift/ADWIN/)
  describes an adaptive drift window. Our baseline instead mixes three fixed
  matching-record window sizes by prequential performance. It is a conventional
  adaptive-window-size baseline, not an ADWIN implementation or reproduction.

This is an extension of a bounded synthetic memory experiment, not a new
Boltzmann generator, peptide model, general intelligence result or full
reproduction of either reference.
