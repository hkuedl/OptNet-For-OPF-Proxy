# OptNet-For-OPF-Proxy
Codes for "OptNet-Embedded Data-Driven Approach for Optimal Power Flow Proxy"
> This work proposed an OptNet-embedded data-driven approach for AC-OPF proxy to provide a feasible solution efficiently. This approach designs a three-stage neural network architecture to represent the OPF problem, where the first stage is used to lift the dimension of the input, the second stage is used to approximate the OPF problem using OptNet, and the third stage is used to decouple the high-dimensional solution to acquire the OPF solution. Finally, to expedite the solving process, a two-step pruning method is proposed to remove the unnecessary inequality constraints and values.

Authors: Yixiong Jia, Yiqin Su, Chenxi Wang, and Yi Wang.


### Data
All the data this paper used can be found in ```4 Bus System/Data``` and ```OPTNET/Data```. 

You can also find the code for processing the data in ```4 Bus System/Data Generate``` and ```Data Generate 14```.

### Reproduction
If you want to run the proposed approach and get the results comparison, you can run ```Results_Show``` in NN-14/NN_ALM-14/DC3-14/OPTNET.

```Model_Compare``` in OPTNET can be used to obtain OPF solutions solved by the pruned model.

## Citation
```
@article{jia2024optnet,
  title={OptNet-Embedded Data-Driven Approach for Optimal Power Flow Proxy},
  author={Jia, Yixiong and Su, Yiqin and Wang, Chenxi and Wang, Yi},
  journal={IEEE Transactions on Industry Applications},
  year={2024},
  publisher={IEEE}
}
