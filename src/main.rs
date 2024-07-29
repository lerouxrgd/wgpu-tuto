use wgpu_tuto::run;

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::new().filter_or(
        "RUST_LOG",
        "wgpu_hal=warn, wgpu_core::device::resource=warn,info",
    ))
    .init();
    pollster::block_on(run())?;
    Ok(())
}
