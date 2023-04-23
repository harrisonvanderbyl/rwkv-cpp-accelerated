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
#include <math.h>
void softmax(float* logits, float* probs, int len)
{
    float sum = 0.0;
    for (int i = 0; i < len; i++)
    {
        probs[i] = exp(logits[i]);
        sum += probs[i];
    }
    for (int i = 0; i < len; i++)
    {
        probs[i] /= sum;
    }
}

float nansum (float* logits, float* probs, int len)
{
    float sum = 0.0;
    for (int i = 0; i < len; i++)
    {
        sum += logits[i] * probs[i];
    }
    return sum;
}

void vecabs(float* logits, float* shifted_logits, int len)
{
    for (int i = 0; i < len; i++)
    {
        shifted_logits[i] = abs(logits[i]);
    }
}

void sub(float* logits, float amount, int len)
{
    for (int i = 0; i < len; i++)
    {
        logits[i] = logits[i] - amount;
    }
}

void copy(float* logits, float* shifted_logits, int len)
{
    for (int i = 0; i < len; i++)
    {
        shifted_logits[i] = logits[i];
    }
}

int* sort(float* shifted_logits, int len)
{
    int* sorted_ids = new int[len];
    for (int i = 0; i < len; i++)
    {
        sorted_ids[i] = i;
    }
    for (int i = 0; i < len; i++)
    {
        for (int j = 0; j < len - 1; j++)
        {
            if (shifted_logits[j] > shifted_logits[j + 1])
            {
                float temp = shifted_logits[j];
                shifted_logits[j] = shifted_logits[j + 1];
                shifted_logits[j + 1] = temp;
                int temp2 = sorted_ids[j];
                sorted_ids[j] = sorted_ids[j + 1];
                sorted_ids[j + 1] = temp2;
            }
        }
    }
    return sorted_ids;
}

int typical(float* logits, float temp = 1.0, float tau = 0.95)
{
    int len = 50277;
    // choose top token
    float max = logits[0];
    int max_id = 0;
    for (int i = 1; i < len; i++)
    {
        if (logits[i] > max)
        {
            max = logits[i];
            max_id = i;
        }
    }
    return max_id;
    // float* probs = new float[len];
    // softmax(logits, probs, len);
    // float ent = nansum(logits, probs, len);
    // float* shifted_logits = new float[len];
    // copy(logits, shifted_logits, len);
    // sub(shifted_logits, ent, len);
    // vecabs(shifted_logits, shifted_logits, len);
    // int* sorted_ids = sort(shifted_logits, len);
    // float* sorted_logits = new float[len];
    // float* sorted_probs = new float[len];
    // for (int i = 0; i < len; i++)
    // {
    //     sorted_logits[i] = shifted_logits[sorted_ids[i]];
    //     sorted_probs[i] = probs[sorted_ids[i]];
    // }
    // float* cumulative_probs = new float[len];
    // float sum = 0.0;
    // for (int i = 0; i < len; i++)
    // {
    //     sum += sorted_probs[i];
    //     cumulative_probs[i] = sum;
    // }
    // int cutoff = 0;
    // for (int i = 0; i < len; i++)
    // {
    //     if (cumulative_probs[i] < tau)
    //     {
    //         cutoff = i;
    //     }
    // }
    // for (int i = 0; i < len; i++)
    // {
    //     if (shifted_logits[i] > sorted_logits[cutoff])
    //     {
    //         probs[i] = 0;
    //     }
    // }
    // if (temp != 1.0)
    // {
    //     for (int i = 0; i < len; i++)
    //     {
    //         probs[i] = pow(probs[i], 1.0 / temp);
    //     }
    // }
    
    // // multinominal sample
    // float sum2 = 0.0;
    // for (int i = 0; i < len; i++)
    // {
    //     sum2 += probs[i];
    // }
    // float r = (float)rand() / (float)RAND_MAX;
    // float sum3 = 0.0;

    // for (int i = 0; i < len; i++)
    // {
    //     sum3 += probs[i] / sum2;
    //     if (sum3 > r)
    //     {
    //         return i;
    //     }
    // }
    // return 0;

    
}