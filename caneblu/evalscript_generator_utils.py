"""
This file contains a function that returns specific Sentinel Hub Evalscripts
based on the supplied script parameter.
"""


def get_evalscript(script, bands=None):
    """
    Returns the appropriate Sentinel Hub Evalscript based on the provided parameter.

    :param script: The script identifier, e.g., 'all_bands', 'ndvi', 'cloud_mask', etc.
    :param bands: (Optional) List of bands to retrieve, used when script='custom_bands'.
    :return: Evalscript as a string.
    """
    if script == "all_bands":
        return """
            function setup() {
                return {
                    input: [{
                        bands: ["B02", "B03", "B04", "B08"]
                    }],
                    output: {
                        bands: 4
                    }
                };
            }

            function evaluatePixel(sample) {
                return [sample.B04, sample.B03, sample.B02, sample.B08];
            }
        """

    elif script == "all_bands_10m":
        return """
                function setup() {
                    return {
                        input: [{
                            bands: ["B02", "B03", "B04", "B08"]
                        }],
                        output: {
                            bands: 4
                        }
                    };
                }

                function evaluatePixel(sample) {
                    return [sample.B04, sample.B03, sample.B02, sample.B08];
                }
            """

    elif script == "fire_risk_bands_60m":
        return """
                function setup() {
                    return {
                        input: [{
                            bands: ["B02", "B03", "B04", "B08", "B11", "B12"]
                        }],
                        output: {
                            bands: 6
                        }
                    };
                }

                function evaluatePixel(sample) {
                    return [sample.B02, sample.B03, sample.B04, sample.B08, sample.B11, sample.B12];
                }
            """

    elif script == "bands_60m_rgb":
        return """
                function setup() {
                    return {
                        input: [{
                            bands: ["B02", "B03", "B04"]
                        }],
                        output: {
                            bands: 3
                        }
                    };
                }

                function evaluatePixel(sample) {
                    return [sample.B02, sample.B03, sample.B04];
                }
            """

    elif script == "bands_60m_ir":
        return """
                function setup() {
                    return {
                        input: [{
                            bands: ["B08", "B11", "B12"]
                        }],
                        output: {
                            bands: 3
                        }
                    };
                }

                function evaluatePixel(sample) {
                    return [sample.B08, sample.B11, sample.B12];
                }
            """

    elif script == "ndvi":
        return """
            function setup() {
                return {
                    input: [{
                        bands: ["B08", "B04"]
                    }],
                    output: {
                        bands: 1
                    }
                };
            }

            function evaluatePixel(sample) {
                let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
                return [ndvi];
            }
        """

    elif script == "custom_bands" and bands:
        input_bands = ", ".join(f'"{band}"' for band in bands)
        output_bands = len(bands)
        band_output = ", ".join(f"sample.{band}" for band in bands)

        return f"""
            function setup() {{
                return {{
                    input: [{{
                        bands: [{input_bands}]
                    }}],
                    output: {{
                        bands: {output_bands}
                    }}
                }};
            }}

            function evaluatePixel(sample) {{
                return [{band_output}];
            }}
        """

    elif script == "cloud_mask":
        return """
            function setup() {
                return {
                    input: [{
                        bands: ["B01", "B09", "B10", "CLM", "CLP"]
                    }],
                    output: {
                        bands: 1
                    }
                };
            }

            function evaluatePixel(sample) {
                return [sample.CLM]; // Cloud mask
            }
        """

    elif script == "false_color":
        return """
            function setup() {
                return {
                    input: [{
                        bands: ["B08", "B04", "B03"]
                    }],
                    output: {
                        bands: 3
                    }
                };
            }

            function evaluatePixel(sample) {
                return [sample.B08, sample.B04, sample.B03];
            }
        """

    else:
        raise ValueError(
            f"Unknown script: {script}. Valid options are 'all_bands', 'ndvi', 'cloud_mask', 'false_color', or 'custom_bands' with a list of bands."
        )
