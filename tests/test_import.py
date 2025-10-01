
def test_import():
    import dp_opt_pricing
    from dp_opt_pricing.config import Config
    assert hasattr(aof, "Config")
    _ = Config()
