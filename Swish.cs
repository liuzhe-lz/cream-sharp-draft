namespace Cream {
    public class Swish : Module
    {
        private bool inplace;

        public Swish(string name, bool inplace = false) : base(name)
        {
            this.inplace = inplace;
        }

        public override Tensor forward(Tensor x)
        {
            if (inplace) {
                return x.mul_(x.sigmoid());
            } else {
                return x.mul(x.sigmoid());
            }
        }
    }
}
