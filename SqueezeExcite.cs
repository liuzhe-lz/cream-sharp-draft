public class SqueezeExcite : DynamicModule
{
    public Module conv_reduce = null;
    public Module act1 = null;
    public Module conv_expand = null;

    public SqueezeExcite(string name) : base(name) { }

    public override Tensor forward(Tensor x)
    {
        x_se = x.mean(new int[] { 2, 3 }, true);
        x_se = conv_reduce.forward(x_se);
        x_se = act1.forward(x_se);
        x_se = conv_expand.forward(x_se);
        return x * gate_fn.forward(x_se);
    }
}
