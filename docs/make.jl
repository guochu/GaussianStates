using Documenter, GaussianStates

makedocs(;
    modules = [GaussianStates],
    sitename = "GaussianStates.jl",
    remotes = nothing,
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = nothing,
        edit_link = nothing,
    ),
    pages = [
        "首页" => "index.md",
        "理论与算法" => "manual.md",
        "API 参考" => "api.md",
    ],
)
