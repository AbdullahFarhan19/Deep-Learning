#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <cmath>
#include <string>
#include <algorithm>
#include <unordered_set>
#include <random>

class Value : public std::enable_shared_from_this<Value>{
    private:
        std::unordered_set<std::shared_ptr<Value>> children;
        std::string op;
        std::string label;
        std::function<void()> _backward; // Basically a function wrapper for lambda functions, needed cuz the compiler needs to know the memory the class occupies before making an object

    public:
        double data;
        double grad; // These are public to closely mimic micrograds functionality

        Value(double data, const std::unordered_set<std::shared_ptr<Value>>& children = {}, std::string op = "", std::string label = "", double grad = 0.0) 
              : data(data), children(children), op(op), label(label), grad(grad){ // Initializer list is the best practice as it prevents the compiler from first initializing instance variables with default values and then overriting them with arguments provided
                _backward = [](){};
            }
        
        // friend functions used because they resolve commutativity issues that a member function will have
        // if it were a member function, the object on the left hand side must be a Value object but we are
        // using shared_ptrs

        friend std::ostream& operator<< (std::ostream& os, const Value& v){ // ostream is the class of cout, this is returned so that cout can chain inputs to print, the friend keyword is like a function that has been given special access to the object and that ignores encapsulation, it takes os and v as parameters where os is cout in cout << v
            os << "Value(data = " << v.data << ", grad = " << v.grad << ")";

            return os;
        }

        friend std::shared_ptr<Value> operator+ (const std::shared_ptr<Value>& self, const std::shared_ptr<Value>& other){ 
            double data = self->data + other->data;

            std::shared_ptr<Value> v = std::make_shared<Value>(data, std::unordered_set<std::shared_ptr<Value>>{self, other}, "+"); 

            Value* vptr = v.get(); // So that the lambda does not catch a shared_ptr resulting in a memory leak
            auto _backward = [vptr, self, other](){
                self->grad += vptr->grad;
                other->grad += vptr->grad;
            };

            v->_backward = _backward;

            return v;
        }

        friend std::shared_ptr<Value> operator+ (double self, const std::shared_ptr<Value>& other){
            return std::make_shared<Value>(self) + other;
        }

        friend std::shared_ptr<Value> operator+ (const std::shared_ptr<Value>& self, double other){
            return self + std::make_shared<Value>(other);
        }

        friend std::shared_ptr<Value> operator* (const std::shared_ptr<Value>& self, const std::shared_ptr<Value>& other){
            double data = self->data * other->data;
            
            std::shared_ptr<Value> v = std::make_shared<Value>(data, std::unordered_set<std::shared_ptr<Value>> {self, other}, "*");

            Value* vptr = v.get();

            auto _backward = [vptr, self, other](){
                self->grad += other->data * vptr->grad;
                other->grad += self->data * vptr->grad;

            };

            v->_backward = _backward;

            return v;

        }

        friend std::shared_ptr<Value> operator* (double self, const std::shared_ptr<Value>& other){
            return std::make_shared<Value>(self) * other;
        }

        friend std::shared_ptr<Value> operator* (const std::shared_ptr<Value>& self, double other){
            return self * std::make_shared<Value>(other);
        }

        friend std::shared_ptr<Value> operator- (const std::shared_ptr<Value>& self){
            return self * (-1);
        }

        friend std::shared_ptr<Value> operator- (const std::shared_ptr<Value>& self, const std::shared_ptr<Value>& other){
            return self + (-other);
        }

        friend std::shared_ptr<Value> operator- (double self, const std::shared_ptr<Value>& other){
            return std::make_shared<Value>(self) - other;
        }

        friend std::shared_ptr<Value> operator- (const std::shared_ptr<Value>& self, double other){
            return self - std::make_shared<Value>(other);
        }

        friend std::shared_ptr<Value> pow (const std::shared_ptr<Value>& self, double power){
            double data = std::pow(self->data, power);

            std::shared_ptr<Value> v = std::make_shared<Value>(data, std::unordered_set<std::shared_ptr<Value>>{self}, "**" + std::to_string(power));

            Value* vptr = v.get();

            auto _backward = [vptr, self, power](){
                self->grad += power * std::pow(self->data, (power - 1)) * vptr->grad;
            };

            v->_backward = _backward;

            return v;
        }

        friend std::shared_ptr<Value> exp (const std::shared_ptr<Value>& self) {
            double data = std::exp(self->data);

            std::shared_ptr<Value> v = std::make_shared<Value>(data, std::unordered_set<std::shared_ptr<Value>>{self}, "e^");

            Value* vptr = self.get();
            auto _backward = [vptr, self](){
                self->grad += vptr->data * vptr->grad;
            };

            self->_backward = _backward;

            return v;
        }

        friend std::shared_ptr<Value> operator/ (const std::shared_ptr<Value>& self, const std::shared_ptr<Value>& other){
            return self * pow(other, -1);
        }

        friend std::shared_ptr<Value> tanh (const std::shared_ptr<Value>& self){
            double data = std::tanh(self->data);

            std::shared_ptr<Value> v = std::make_shared<Value>(data, std::unordered_set<std::shared_ptr<Value>>{self}, "tanh");

            Value* vptr = v.get();

            auto _backward = [vptr, self](){
                self->grad += (1 - std::pow(vptr->data, 2)) * vptr->grad; //Derivative of tanhx is sech square x which is equal to 1 - tanh square x
            };

            v->_backward = _backward;

            return v;
        }

    
        void backward (){
            std::vector<std::shared_ptr<Value>> topo = {};
            std::unordered_set<std::shared_ptr<Value>> visited = {}; // Unordered set used to avoid O(N) lookup

            std::function<void(const std::shared_ptr<Value>&)> build_topo = [&](const std::shared_ptr<Value>& node){ // You cannot bind a temporary object to a non-const reference
                if(visited.find(node) == visited.end()){
                    visited.insert(node);

                    for(std::shared_ptr<Value> n : node->children){
                        build_topo(n);
                    }

                    topo.push_back(node);
                }
            };

            build_topo(shared_from_this());
            this->grad = 1.0;
            

            for(int i = topo.size() - 1; i >= 0; i--){
                topo[i]->_backward();
            }
            
        }
        
};

class Neuron {
    private:
        std::vector<std::shared_ptr<Value>> w;
        std::shared_ptr<Value> b;
    public:
        Neuron(int nin){
            static std::mt19937 engine(std::random_device{}()); // Uses the Mersene Twister algorithm that is much more "random" then the rand() function and doesnt have the RAND_MAX constraint
            static std::uniform_real_distribution<double> ran(-1,1); // They are both static so they are not created again and again

            for(int i = 0; i < nin; i++){
                double rw = ran(engine);
                w.push_back(std::make_shared<Value>(rw));
            }

            b = std::make_shared<Value>(ran(engine));
        }

        std::shared_ptr<Value> operator()(const std::vector<std::shared_ptr<Value>>& x) { // This is a vector of shared_ptrs because the input to every neuron in a series of layers will be the output of the previous layer and output of the previous layer is a vector of shared ptrs
            std::shared_ptr<Value> sum = b;

            for(int i = 0; i < x.size(); i++){
                sum = sum + w[i] * x[i];
                
            }

            return tanh(sum);

        }

        std::vector<std::shared_ptr<Value>> parameters(){
            std::vector<std::shared_ptr<Value>> out = w;
            out.push_back(b);

            return out;
        }

};

class Layer{
    private:
        std::vector<std::shared_ptr<Neuron>> neurons;

    public:
        Layer(int nin, int nn){
            for(int i = 0; i < nn; i++){
                neurons.push_back(std::make_shared<Neuron>(nin));
            }
        }

        std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& x){
            std::vector<std::shared_ptr<Value>> outs;

              for(std::shared_ptr<Neuron> neuron : neurons){
                outs.push_back((*neuron)(x));
              }   
              
            return outs;
        }

        std::vector<std::shared_ptr<Value>> parameters(){
            std::vector<std::shared_ptr<Value>> out;

            for(const auto& neuron : neurons){
                std::vector<std::shared_ptr<Value>> params = neuron->parameters(); 

                out.insert(out.end(), params.begin() ,params.end());
            }

            return out;
        }
};

class MLP{
    private:
        std::vector<std::shared_ptr<Layer>> layers;

    public:
        MLP(int nin, std::vector<int>& nns){
            std::vector<int> dim = nns;
            dim.insert(dim.begin(), nin); // [nin, nns[0], nns[1], ...] 

            for(int i = 0; i < dim.size() - 1; i++){
                layers.push_back(std::make_shared<Layer>(dim[i], dim[i+1]));
            }

        }

        std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& x){
            std::vector<std::shared_ptr<Value>> xcpy = x;

            for(std::shared_ptr<Layer> layer : layers){
                xcpy = (*layer)(xcpy);
            }

            return xcpy;
        }

        std::vector<std::shared_ptr<Value>> parameters(){
            std::vector<std::shared_ptr<Value>> out;

            for(const auto& layer : layers){ // const auto& prevents copying
                std::vector<std::shared_ptr<Value>> params = layer->parameters();

                out.insert(out.end(), params.begin(), params.end());
            }

            return out;
        }
};

