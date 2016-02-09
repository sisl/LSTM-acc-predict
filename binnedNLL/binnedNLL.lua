require 'nn'
local binnedNLL, parent = torch.class('binnedNLL', 'nn.Criterion')

function binnedNLL:__init(n, mu, sigma)
    parent.__init(self)
    n = n or 1
    mu = mu or 0
    sigma = sigma or 0
    self.n = n
    self.mu = mu
    self.sigma = sigma
end

-- -- Compute pdf at target for normal dist w/ mean mu and std dev s
-- local function normal(target, mu, s)
--     if type(mu) == 'number' then
--         local arg = -torch.pow(target - mu, 2)/(2*s^2)
--         local exparg = torch.exp(arg)
--         return exparg/(math.sqrt(2*math.pi)*s)
--     else
--         local arg = torch.cdiv(-torch.pow(target - mu, 2), torch.pow(s, 2)*2)
--         local exparg = torch.exp(arg)
--         return torch.div(torch.cdiv(exparg, s), math.sqrt(2*math.pi))
--     end
-- end

-- -- Function to calculate quantity pi that is present in gradients
-- local function getPi(target, input, self)
--     local num = torch.Tensor(input:size(1), self.n)
--     local pi = torch.Tensor(#num)

--     for i = 1, self.n do

--         if self.mu == 0 then
--             local w = input[{{}, i}]
--             local mu = input[{{}, self.n + i}]
--             local s = input[{{}, 2*self.n + i}]
--             num[{{}, i}] = torch.cmul(w, normal(target, mu, s))
--         else
--             local w = torch.exp(input[{{}, i}])
--             num[{{}, i}] = torch.cmul(w, normal(target, self.mu[i], self.sigma[i]))
--         end
--     end

--     for i = 1, input:size(1) do
--         pi[{i, {}}] = num[i]:div(torch.sum(num[i]))
--     end
--     return pi
-- end

-- -- Gradient wrt mu for normal dist
-- local function grad_mu(target, mu, s)
--     return torch.cdiv(mu - target, torch.pow(s, 2))
-- end

-- -- Gradient wrt sigma for normal dist
-- local function grad_s(target, mu, s)
--     local g1 = -torch.cdiv(torch.pow(target - mu, 2), torch.pow(s, 3))
--     local g2 = torch.cdiv(torch.ones(s:size()), s)
--     return g1 + g2
-- end

-- -- Computes sum of Gaussian components
-- local function getSum(input, target, n)
--     local sum = torch.zeros(input:size(1))
--         for i = 1, n do
--             local w = input[{{}, i}]
--             local mu = input[{{}, n + i}]
--             local s = input[{{}, 2*n + i}]
--             sum = sum + w * normal(target, mu, s)
--         end
--     return sum
-- end

-- Returns loss
function normalNLL:updateOutput(input, target)
    if self.mu == 0 then
        -- Single normal distribution
        if self.n == 1 and input:size(2) == 2 then
            local mu = input[{{}, 1}] -- mean
            local s = input[{{}, 2}] -- std dev
            self.output = torch.cmin(-torch.log(normal(target, mu, s)), 10)

        elseif self.n > 1 and input:size(2) == 3*self.n then
            self.output = -torch.log(getSum(input, target, self.n))
        else
            error('Invalid number of inputs')
        end
    else
        if self.n > 1 and input:size(2) == self.n then
            local sum = torch.zeros(input:size(1))
            for i = 1, self.n do
                local w = torch.exp(input[{{}, i}])
                sum = sum + torch.cmul(w, normal(target, self.mu[i], self.sigma[i]))
            end
            self.output = -torch.log(sum)
        else
            error('Invalid number of inputs')
        end
    end
    return torch.mean(self.output)
end

-- Returns gradients
function normalNLL:updateGradInput(input, target)
    self.gradInput:resizeAs(input)
    self.gradInput:zero()

    if self.n == 1 then

        local mu = input[{{}, 1}] -- mean
        local s = input[{{}, 2}] -- std dev

        -- Gradient wrt mu
        self.gradInput[{{}, 1}] = grad_mu(target, mu, s)

        -- Gradient wrt sigma
        self.gradInput[{{}, 2}] = grad_s(target, mu, s)

    else
        -- Calculate pi
        local pi = getPi(target, input, self)

        if self.mu == 0 then 

            -- Extract various components of input
            local w = input[{{}, {1, self.n}}]
            local mu = input[{{}, {self.n + 1, 2*self.n}}]
            local s = input[{{}, {2*self.n + 1, 3*self.n}}]

            -- Weight gradients:
            self.gradInput[{{}, {1, self.n}}] = -torch.cdiv(pi, w)

            for i = 1, self.n do
                -- Mean gradients
                self.gradInput[{{}, self.n + i}] = torch.cmul(pi[{{}, i}], torch.cdiv(mu[{{}, i}] - target, torch.pow(s[{{}, i}], 2)))

                -- Std dev gradients
                self.gradInput[{{}, 2*self.n + i}] = torch.cdiv(pi[{{}, i}], s[{{}, i}]) - 
                torch.cmul(pi[{{}, i}], torch.cdiv(torch.pow(target - mu[{{}, i}], 2), torch.pow(s[{{}, i}], 3)))
            end
        else
            local w = torch.exp(input[{{}, {1, self.n}}])

            -- Weight gradients:
            self.gradInput[{{}, {1, self.n}}] = -pi
        end
    end

    -- Clip extreme gradient values
    self.gradInput = torch.cmin(self.gradInput, 5)
    self.gradInput = torch.cmax(self.gradInput, -5)
    return self.gradInput
end
