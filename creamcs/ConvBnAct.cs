using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public class ConvBnAct : Module
{
    private Module conv;
    private Module bn1;
    private Module act1;

    public ConvBnAct(string name, dynamic arch) : base(name)
    {
        conv = Loader.LoadArch(arch.conv);
        bn1 = Loader.LoadArch(arch.bn1);
        act1 = Loader.LoadArch(arch.act1);

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        x = conv.forward(x);
        x = bn1.forward(x);
        x = act1.forward(x);
        return x;
    }
}
