package llm

type LoadOptions struct {
	NumCtx    int
	GPULayers int
	Threads   int
}

func DefaultLoadOptions() *LoadOptions {
	return &LoadOptions{
		NumCtx:    2048,
		GPULayers: 0,
		Threads:   0,
	}
}
