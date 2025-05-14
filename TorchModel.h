#pragma once
#include <torch/torch.h>

class TorchModel : public torch::nn::Module {
   public:
    virtual ~TorchModel() = default;

    virtual torch::Tensor forward(torch::Tensor input) = 0;
};
