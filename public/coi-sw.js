// GitHub Pages cannot send COOP/COEP headers, and without them the page is not
// cross-origin isolated, so ONNX Runtime falls back to a single WASM thread.
// This service worker re-serves the site's own files with those headers.
// Cross-origin requests (the model download from Hugging Face) are left alone:
// they are CORS fetches and pass COEP as they are.

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()));

self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (new URL(request.url).origin !== self.location.origin) return;

  event.respondWith(
    fetch(request).then((response) => {
      if (response.status === 0) return response;
      const headers = new Headers(response.headers);
      headers.set('Cross-Origin-Opener-Policy', 'same-origin');
      headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
      headers.set('Cross-Origin-Resource-Policy', 'same-origin');
      return new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers,
      });
    }),
  );
});
