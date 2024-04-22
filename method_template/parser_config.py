from __future__ import annotations

from nerfstudio.plugins.registry_dataparser import DataParserSpecification

from method_template.template_dataparser import TemplateDataParserConfig

XrayDataparser = DataParserSpecification(config=TemplateDataParserConfig())
