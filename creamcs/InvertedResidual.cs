using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public class InvertedResidual : Module
{
    private Module conv_pw;
    private Module bn1;
    private Module act1;
    private Module conv_dw;
    private Module bn2;
    private Module act2;
    private Module se;
    private Module conv_pwl;
    private Module bn3;

    public InvertedResidual(string name, dynamic arch) : base(name)
    {
        conv_pw = Loader.LoadArch(arch.conv_pw);
        bn1 = Loader.LoadArch(arch.bn1);
        act1 = Loader.LoadArch(arch.act1);
        conv_dw = Loader.LoadArch(arch.conv_dw);
        bn2 = Loader.LoadArch(arch.bn2);
        act2 = Loader.LoadArch(arch.act2);
        se = Loader.LoadArch(arch.se);
        conv_pwl = Loader.LoadArch(arch.conv_pwl);
        bn3 = Loader.LoadArch(arch.bn3);

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        x = conv_pw.forward(x);
        x = bn1.forward(x);
        x = act1.forward(x);

        x = conv_dw.forward(x);
        x = bn2.forward(x);
        x = act2.forward(x);

        x = se.forward(x);

        x = conv_pwl.forward(x);
        x = bn3.forward(x);

        // FIXME: residual is not implemented because it needs shape inference

        return x;
    }
}
