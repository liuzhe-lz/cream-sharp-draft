namespace Cream {
    public class SelectAdaptivePool2d : Module
    {
        private int output_size;
        private Module pool;
        private bool flatten;

        public SelectAdaptivePool2d(string name, int output_size = 1, string pool_type = "fast", bool flatten = false)
            : base(name)
        {
            this.output_size = output_size;
            this.flatten = flatten;

            if (pool_type == "") {
                pool = Identity();
            } else if (pool_type == "avg") {
                pool = AdaptiveAvgPool2d(output_size);
            } else {
                throw new ArgumentException("pool_type not yet implemented", pool_type);
            }

            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            x = pool.forward(x);
            if (flatten) {
                x = x.flatten(1);
            }
            return x;
        }
    }
}
