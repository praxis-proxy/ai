import { Plugin } from "@opencode/plugin"

export default Plugin.define({
  id: "praxis-auth",
  async setup(ctx) {
    await ctx.session.hook(
      "model.request",
      (event) => {
        const trustedURL = process.env.PRAXIS_BASE_URL ?? "http://127.0.0.1:8080/v1"
        if (event.baseURL !== trustedURL) {
          throw new Error("Praxis baseURL does not match the trusted gateway")
        }
        const password = process.env.GATEWAY_AUTH_PASSWORD
        if (!password) throw new Error("GATEWAY_AUTH_PASSWORD is required for Praxis")
        const token = Buffer.from(`gateway:${password}`).toString("base64")
        event.headers.Authorization = `Basic ${token}`
      },
      { providerID: "praxis" },
    )
  },
})
