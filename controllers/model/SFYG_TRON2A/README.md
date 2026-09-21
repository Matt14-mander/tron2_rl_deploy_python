# SFYG model contract

Place the WholeBody artifacts here:

- `encoder.onnx`: one input `420`, one output `3`;
- `policy.onnx`: one input `78`, one output `10`.

The policy input order is `encoder(3) + proprioception(42) + normalized future
wrench(30) + base velocity command(3)`. Do not copy the ordinary SFYG/SF
`48 -> 10` policy into this directory.
