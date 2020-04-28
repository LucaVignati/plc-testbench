# AlohaECC

## Basic Usage

```python
    from alohaecc.ecc_testbench import ecc_testbench
    from alohaecc.ecc_algorithm import zeros_ecc, last_packet_ecc

    ecc_algorithm = last_packet_ecc
    testbench = ecc_testbench(ecc_algorithm)

    wave_file = '/folder/input_file.wav'
    mean_square_error = testbench.run(wave_file)

```
## Unit Testing

```bash
    python -m unittest test.test_ecc_testbench.test_ecc_testbench -v
```