// def typical(logits, temp=1.0, tau=0.95, **kwargs):
//         import torch
//     # do it in pytorch
//         import numpy as np
//         probs = torch.nn.functional.softmax(logits.float(), dim=-1)
//         logits = -torch.log(probs)
//         ent = torch.nansum(logits * probs, dim=-1, keepdim=True)
//         shifted_logits = torch.abs(logits - ent)
//         sorted_ids = torch.argsort(shifted_logits)
//         sorted_logits = shifted_logits[sorted_ids]
//         sorted_probs = probs[sorted_ids]
//         cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
//         cutoff = np.sum(cumulative_probs < tau)
//         probs[shifted_logits > sorted_logits[cutoff]] = 0
//         if temp != 1.0:
//             probs = probs ** (1.0 / temp)
//         out = torch.multinomial(probs, num_samples=1)[0]
//         return int(out)

#include "NumCpp.hpp"
int typical(float* _logits, float _temp = 0.9, float _tau = 0.8)
{
    int len = VOCAB;
    // choose top token
    nc::NdArray<double> logits = nc::NdArray<double>(1,len);
    for (int i = 0; i < len; i++) {
        
        logits[i] = _logits[i];
    }
    // unsigned long long VOCAB = 65536;
    nc::NdArray<double> probs = nc::special::softmax(logits); 
    logits = -nc::log(probs);
    nc::NdArray<double> ent = nc::nansum(logits * probs);
    nc::NdArray<double> shifted_logits = nc::abs(logits - ent);
    nc::NdArray<uint32_t> sorted_ids = nc::argsort(shifted_logits);
    nc::NdArray<double> sorted_logits = shifted_logits[sorted_ids];
    nc::NdArray<double> sorted_probs = probs[sorted_ids];
    nc::NdArray<double> cumulative_probs = nc::cumsum(sorted_probs);
    nc::NdArray<double> tau = nc::NdArray<double>(1,1);
    tau[0] = _tau;
    auto mask = (cumulative_probs < tau);
    // convert mask to int
    nc::NdArray<int> mask_int = nc::NdArray<int>(1,mask.size());
    for (int i = 0; i < mask.size(); i++) {
        mask_int[i] = mask[i];
    }

    // get cutoff
    auto cutoff = nc::sum(mask_int);
    // set probs to 0
    probs[shifted_logits > sorted_logits[cutoff]] = 0;
    if (_temp != 1.0) {
        probs = nc::power(probs, 1.0 / _temp);
    }

    // get random token
    auto out = nc::random::discrete<int>(nc::shape(tau),probs);
    return out[0];  
}

std::vector<unsigned long long> typical(int batchsize, float* _logits, float _temp = 0.9, float _tau = 0.8){
    std::vector<unsigned long long> out;
    for(int i = 0; i < batchsize; i++){
        out.push_back(typical(&_logits[i*VOCAB], _temp, _tau));
    }
    return out;
}