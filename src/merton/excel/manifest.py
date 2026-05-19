"""Office.js add-in manifest XML generation.

Office.js (Excel for the web / Mac / Windows M365) sideloads add-ins via an
XML manifest that describes the add-in's identity, permissions, and the
URLs to the JavaScript / functions / settings files. We generate this
manifest at install time so it points at the user's locally-running
``merton excel server``.

The manifest schema is documented at
https://learn.microsoft.com/en-us/javascript/api/manifest .
"""

from __future__ import annotations

import uuid
from pathlib import Path

_MANIFEST_TEMPLATE = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<OfficeApp
    xmlns="http://schemas.microsoft.com/office/appforoffice/1.1"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xmlns:bt="http://schemas.microsoft.com/office/officeappbasictypes/1.0"
    xmlns:ov="http://schemas.microsoft.com/office/taskpaneappversionoverrides"
    xsi:type="TaskPaneApp">
  <Id>{add_in_id}</Id>
  <Version>{version}</Version>
  <ProviderName>merton</ProviderName>
  <DefaultLocale>en-US</DefaultLocale>
  <DisplayName DefaultValue="merton" />
  <Description DefaultValue="Merton structural credit-risk model — Excel custom functions." />
  <IconUrl DefaultValue="{base_url}/static/icon-32.png" />
  <HighResolutionIconUrl DefaultValue="{base_url}/static/icon-80.png" />
  <SupportUrl DefaultValue="https://merton.readthedocs.io" />
  <AppDomains>
    <AppDomain>{base_url}</AppDomain>
  </AppDomains>
  <Hosts>
    <Host Name="Workbook" />
  </Hosts>
  <DefaultSettings>
    <SourceLocation DefaultValue="{base_url}/taskpane.html" />
  </DefaultSettings>
  <Permissions>ReadWriteDocument</Permissions>
  <VersionOverrides xmlns="http://schemas.microsoft.com/office/taskpaneappversionoverrides" xsi:type="VersionOverridesV1_0">
    <Hosts>
      <Host xsi:type="Workbook">
        <Runtimes>
          <Runtime resid="JsRuntime.Url" lifetime="long" />
        </Runtimes>
        <AllFormFactors>
          <ExtensionPoint xsi:type="CustomFunctions">
            <Script>
              <SourceLocation resid="JsRuntime.Url" />
            </Script>
            <Page>
              <SourceLocation resid="Functions.Page.Url" />
            </Page>
            <Metadata>
              <SourceLocation resid="Functions.Metadata.Url" />
            </Metadata>
            <Namespace resid="Functions.Namespace" />
          </ExtensionPoint>
        </AllFormFactors>
      </Host>
    </Hosts>
    <Resources>
      <bt:Urls>
        <bt:Url id="JsRuntime.Url" DefaultValue="{base_url}/static/functions.js" />
        <bt:Url id="Functions.Page.Url" DefaultValue="{base_url}/static/functions.html" />
        <bt:Url id="Functions.Metadata.Url" DefaultValue="{base_url}/functions.json" />
      </bt:Urls>
      <bt:ShortStrings>
        <bt:String id="Functions.Namespace" DefaultValue="MERTON" />
      </bt:ShortStrings>
    </Resources>
  </VersionOverrides>
</OfficeApp>
"""


def deterministic_add_in_id(base_url: str) -> str:
    """Generate a stable UUID derived from ``base_url`` so re-installing the
    add-in for the same server URL keeps the same identity."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, base_url))


def render_manifest(
    *,
    base_url: str = "http://localhost:8000",
    version: str = "1.0.0.0",
    add_in_id: str | None = None,
) -> str:
    """Render the Office.js add-in manifest XML.

    Parameters
    ----------
    base_url
        Where the merton excel server is reachable (default
        ``http://localhost:8000``).
    version
        Four-part add-in version (Microsoft requires this exact format).
    add_in_id
        Optional override. Defaults to a deterministic UUID derived from
        ``base_url`` so the same server URL always produces the same
        manifest identity.
    """
    if add_in_id is None:
        add_in_id = deterministic_add_in_id(base_url)
    return _MANIFEST_TEMPLATE.format(
        add_in_id=add_in_id,
        version=version,
        base_url=base_url.rstrip("/"),
    )


def write_manifest(
    path: str | Path, *, base_url: str = "http://localhost:8000", version: str = "1.0.0.0"
) -> Path:
    """Write the manifest to disk and return the resulting :class:`Path`."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_manifest(base_url=base_url, version=version))
    return p


__all__ = ["deterministic_add_in_id", "render_manifest", "write_manifest"]
