import torch

# output��ģ�͵��������ģ�ͶԲ�ͬ�������֡�shape: [batch_size, num_classes]

# target����ʵ������ǩ��shape: [batch_size, ]

# topk����Ҫ����top_k׼ȷ���е�kֵ��Ԫ�����͡�Ĭ��Ϊ(1, 5)������������top1��top5�ķ���׼ȷ��

"""
torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)
input (Tensor) �C ��������

k(int) �C ��top-k���е�k

dim (int, optional) �C �����ά,�ظ���dimά�ȷ�����������input�� k �����ֵ�������ָ��dim����Ĭ��Ϊinput�����һά��

largest (bool, optional) �C ����ֵ�����Ʒ���������Сֵ,True���ֵ

sorted (bool, optional) �C ����ֵ�����Ʒ���ֵ�Ƿ�����

out (tuple, optional) �C ��ѡ������� (Tensor, LongTensor) output buffer

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


"������ģ�������batch_size��num_of_class����Ŀ��label��num_of_class��������Ԫ�飨�ֱ�����top����"
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
# input����������
# k��ָ�����ص�ǰ��λ��ֵ
# dim�������ά��
# largest���������ֵ
# sorted������ֵ�Ƿ�����
# out����ѡ�������


if __name__ == '__main__':
    output = torch.randint(low=0, high=6, size=[8, 10])
    target = torch.ones(8, dtype=torch.long)
    print(accuracy(output, target))