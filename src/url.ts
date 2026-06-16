// Server-profile selection + URL wiring

/**
 * The server_profile this session requests, or undefined for the server
 * default. 
 * 
 * `enableVideo === false` selects "audio_only" 
 * "default" is no-op
 */
export function effectiveServerProfile(
  serverProfile: string | undefined,
  enableVideo: boolean,
): string | undefined {
  const prof = serverProfile != null ? serverProfile : enableVideo ? undefined : "audio_only";
  return prof && prof !== "default" ? prof : undefined;
}

/**
 * Direct ws(s):// mode for applying server_profile to the URL query
 */
export function applyServerProfileToWsUrl(
  url: string,
  serverProfile: string | undefined,
  enableVideo: boolean,
): string {
  const profile = effectiveServerProfile(serverProfile, enableVideo);
  if (!profile) return url;
  // inferred profile defers to a server_profile already in the URL
  if (serverProfile == null && new URL(url).searchParams.has("server_profile")) {
    return url;
  }
  const u = new URL(url);
  u.searchParams.set("server_profile", profile);
  return u.toString();
}

/**
 * Broker mode — the JSON body for POST /allocate
 * 
 * Returns undefined when no profile is selected
 */
export function allocateBody(
  serverProfile: string | undefined,
  enableVideo: boolean,
): string | undefined {
  const profile = effectiveServerProfile(serverProfile, enableVideo);
  return profile ? JSON.stringify({ server_profile: profile }) : undefined;
}
