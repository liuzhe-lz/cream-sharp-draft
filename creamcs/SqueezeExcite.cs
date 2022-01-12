using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public class SqueezeExcite : Module
{
    private Module avg_pool;
    private Module conv_reduce;
    private Module act1;
    private Module conv_expand;

    public SqueezeExcite(string name, dynamic arch) : base(name)
    {
        avg_pool = Loader.LoadArch(arch.avg_pool);
        conv_reduce = Loader.LoadArch(arch.conv_reduce);
        act1 = Loader.LoadArch(arch.act1);
        conv_expand = Loader.LoadArch(arch.conv_expand);

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        x = avg_pool.forward(x);
        x = conv_reduce.forward(x);
        x = act1.forward(x);
        x = conv_expand.forward(x);

        // FIXME: asserting gate_fn is sigmoid
        return x * x.sigmoid();
    }
}
