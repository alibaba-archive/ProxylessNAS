from blocks_cifar import get_blocks
import torch
blks = get_blocks(face=True)
#measure(blks, cpu=True)
result_path = 'architecture_parameter.txt'
f = open(result_path, 'w')
for b in blks:
    if isinstance(b, list):
        for net in b:
            #input = torch.randn(input_shape)
            f.write('%.7f ' % 1)
        f.write('\n')
f.close()