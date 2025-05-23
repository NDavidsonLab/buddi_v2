# Development Notes

## What's done:
1. Preprocessing module under `budd_v2.preprocessing.sc_augmentor`, which does:
    - Random uniform pseudobulk generation
    - Realistic pseudobulk generation
    - Cell type dominant pseudobulk generation
    - Some control over noise injection
    - Visualization of Integrated pseudobulk and bulk data with `buddi_v2.plotting.plot_data`
2. `BuDDI3`/`BuDDI4` class that works with:
    - generic data class `buddi_v2.data.BuDDINData`
    - generic fitter function `buddi_v2.model.fit.fit_buddi`
    - generic visualization `buddi_v2.plotting.plot_latent_space` and `buddi_v2.plotting.plot_loss`
    - dataset classes `buddi_v2.dataset.buddi3_dataset.*` and `buddi_v2.dataset.buddi4_dataset.*`
3. Some training Examples

## Missing:
- Generic BuDDI with arbitrary number of encoder branches (which can be simply implemented by subclassing `buddi_v2.model.buddi_abstract_class`), but will need custom dataset classes for use with generic fitter function
- BuDDI validation helper (currently in analysis repo)
- Installation