def dice_loss(m1, m2):
    num = m1.size(0)
    m1  = m1.view(num,m1.size(1),-1)
    m2  = m2.view(num,m1.size(1),-1)
    intersection = (m1 * m2)
    scores = (2. * intersection.sum(2)+1) / (m1.sum(2) + m2.sum(2)+1)
    score = scores.mean()
    return 1-score


# def dice_score(m1, m2):
#     num = m1.size(0)
#     m1  = m1.view(num,-1)
#     m2  = m2.view(num,-1)
#     intersection = (m1 * m2)
#     scores = (2. * intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
#     # score = scores.sum()/num
#     return scores.sum()

if __name__ == '__main__':
    import torch
    torch.manual_seed(1)
    x = torch.rand((4, 1, 128, 128))
    y = torch.rand((4, 1, 128, 128))
    print(dice_loss(x,y))