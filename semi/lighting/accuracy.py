import torch

# output：模型的输出，即模型对不同类别的评分。shape: [batch_size, num_classes]

# target：真实的类别标签。shape: [batch_size, ]

# topk：需要计算top_k准确率中的k值，元组类型。默认为(1, 5)，即函数返回top1和top5的分类准确率

"""
torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)
input (Tensor) – 输入张量

k(int) – “top-k”中的k

dim (int, optional) – 排序的维,沿给定dim维度返回输入张量input中 k 个最大值。如果不指定dim，则默认为input的最后一维。

largest (bool, optional) – 布尔值，控制返回最大或最小值,True最大值

sorted (bool, optional) – 布尔值，控制返回值是否排序

out (tuple, optional) – 可选输出张量 (Tensor, LongTensor) output buffer

"""

def accuracy(output, target, topk=(1, 5)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k / batch_size)
    return res


"输入是模型输出（batch_size×num_of_class），目标label（num_of_class向量），元组（分别向求top几）"
def accu(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)
# input：输入张量
# k：指定返回的前几位的值
# dim：排序的维度
# largest：返回最大值
# sorted：返回值是否排序
# out：可选输出张量


if __name__ == '__main__':
    output = torch.randint(low=0, high=6, size=[8, 10])
    target = torch.ones(8, dtype=torch.long)
    print(accuracy(output, target))