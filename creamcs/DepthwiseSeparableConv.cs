using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public class DepthwiseSeparableConv : Module
{
    private Module conv_dw;
    private Module bn1;
    private Module act1;
    private Module se;
    private Module conv_pw;
    private Module bn2;
    private Module act2;

    public DepthwiseSeparableConv(string name, dynamic arch) : base(name)
    {
        conv_dw = Loader.LoadArch(arch.conv_dw);
        bn1 = Loader.LoadArch(arch.bn1);
        act1 = Loader.LoadArch(arch.act1);
        se = Loader.LoadArch(arch.se);
        conv_pw = Loader.LoadArch(arch.conv_pw);
        bn2 = Loader.LoadArch(arch.bn2);
        act2 = Loader.LoadArch(arch.act2);

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        x = conv_dw.forward(x);
        x = bn1.forward(x);
        x = act1.forward(x);

        x = se.forward(x);

        x = conv_pw.forward(x);
        x = bn2.forward(x);
        x = act2.forward(x);

        // FIXME: residual is not implemented because it needs shape inference

        return x;
    }
}
