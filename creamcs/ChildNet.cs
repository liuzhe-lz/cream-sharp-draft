using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public class ChildNet : Module
{
    private Module conv_stem;
    private Module bn1;
    private Module act1;
    private Module blocks;
    private Module global_pool;
    private Module conv_head;
    private Module act2;
    private Module classifier;
    private Module flatten;

    public ChildNet(string name, dynamic arch) : base(name)
    {
        conv_stem = Loader.LoadArch(arch.conv_stem);
        bn1 = Loader.LoadArch(arch.bn1);
        act1 = Loader.LoadArch(arch.act1);
        blocks = Loader.LoadArch(arch.blocks);
        global_pool = Loader.LoadArch(arch.global_pool);
        conv_head = Loader.LoadArch(arch.conv_head);
        act2 = Loader.LoadArch(arch.act2);
        classifier = Loader.LoadArch(arch.classifier);
        flatten = Flatten(1);

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        // forward_features
        x = conv_stem.forward(x);
        x = bn1.forward(x);
        x = act1.forward(x);
        x = blocks.forward(x);
        x = global_pool.forward(x);
        x = conv_head.forward(x);
        x = act2.forward(x);

        x = flatten.forward(x);
        // FIXME: dropout is a little tricky because we can't set it in constructor
        return classifier.forward(x);
    }
}
