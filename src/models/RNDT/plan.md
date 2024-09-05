### 3 Big things to consider
- spikes input: all spikes, part of the spikes, masked spikes
- query
  - region query: all regions in this session, all regions in all sessions, random regions
  - neuron query
- loss:
  - region-wise linear decoder equals neuron-wise linear decoder, which will cause too many parameters. Any solutions?
    - 1) Use neuron-wise linear decoder first. (probably in a single session)
      2) Try to find some patterns in these linear matrixes, like low-ranking, sparsity, etc.
      3) Use these patterns to reduce the computational complexity.
  - 2 training objectives (maybe 2 stages): simply extract region features (no masking) (train all parameters), use masking
    (or other methods mentioned above) and train only the transformer parameters (maybe freeze the region-wise linear decoder parameters).



### Notes
- We assume region queries within each batch are the same.
- We assume padding/masking within each batch are the same.
- The final region-wise linear decoder is actually not that important. The most valuable part is the region representation
  before being linearly decoded.


### TODO:
- Check if max_F will break in iTransformer mode.