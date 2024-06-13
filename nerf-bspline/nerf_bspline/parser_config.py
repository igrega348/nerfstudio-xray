from __future__ import annotations

from nerfstudio.plugins.registry_dataparser import DataParserSpecification

from nerf_bspline.template_dataparser import TemplateDataParserConfig

XrayDataParser = DataParserSpecification(config=TemplateDataParserConfig())
