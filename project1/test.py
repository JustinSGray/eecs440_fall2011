import unittest

from proj1 import Node, load_project_data

from shell.mldata import ExampleSet

class Test(unittest.TestCase): 
    
    def test_node_shape(self): 
    
        tree = Node()
        children = {1:Node(),2:Node()}
        tree.children = children
        self.assertEqual((3,1),tree.shape())
        
        tree.children[1].children = children = {3:Node(),4:Node()}
        self.assertEqual((5,2),tree.shape())
        
        tree.children[1].children[3].children = children = {5:Node(),6:Node()}
        self.assertEqual((7,3),tree.shape())
        
    def test_load_proj_data(self): 
        train_data, test_data = load_project_data('example')
        
        self.assertEqual(7,len(train_data))
        self.assertEqual(2,len(test_data))
        
    def test_is_partable_data(self): 
        train_data, test_data = load_project_data('example')
        
        examples = [ex for ex in train_data] + [ex for ex in test_data]
        data = ExampleSet(examples) 
        
        n = Node()   
        
        
        self.assertTrue(n.check_ex_set(examples,[1,3]))  
        self.assertTrue(n.check_ex_set(examples,[1,3]))
        
        index,part_data = n.max_GR(examples,[1,3])  
        self.assertEqual(index,1)
        test_part_data = [ex for ex in examples if ex[1]=='red']
        self.assertEqual(set(test_part_data),set(part_data['red']))
        
        self.assertTrue(n.check_ex_set(part_data['red'],[3,])[1])
        index,sub_data = n.max_GR(part_data['red'],[3])  
        self.assertEqual(index,3)
        
        test_part_data = [ex for ex in examples if ex[1]=='green']
        self.assertEqual(set(test_part_data),set(part_data['green']))
        self.assertFalse(n.check_ex_set(part_data['green'],[3,])[1])
        
        test_part_data = [ex for ex in examples if ex[1]=='blue']
        self.assertEqual(set(test_part_data),set(part_data['blue']))
        self.assertTrue(n.check_ex_set(part_data['blue'],[3,])[1])
        index,sub_data = n.max_GR(part_data['blue'],[3])  
        self.assertEqual(index,3)
        
        index,sub_data = n.max_GR(examples,[3,1])  
        self.assertEqual(index,3)
        
            
        
    def test_part_discrete_data(self): 
        train_data, test_data = load_project_data('example')
        
        examples = [ex for ex in train_data] + [ex for ex in test_data]
        data = ExampleSet(examples)    
        
        n = Node()
        
        H_x,H_y_x, part_data = n.partition_data(data,1)
        
        self.assertAlmostEqual(0.61219,H_y_x,3)
        self.assertAlmostEqual(1.5849,H_x,3)
        
        H_x,H_y_x, part_data  = n.partition_data(data,3)
        self.assertAlmostEqual(0.61219,H_y_x,3)
        self.assertAlmostEqual(1.5849,H_x,3)
        
        attr_index,part_data = n.max_GR(examples,[1,3])
        self.assertEqual(attr_index,1)
        
        part_data_test = {}
        for ex in examples:
            part_data_test.setdefault(ex[1],[]).append(ex) 
        self.assertEqual(part_data_test,part_data)    
        
        attr_index,part_data = n.max_GR(examples,[3,1])
        self.assertEqual(attr_index,3)
        
        part_data_test = {}
        for ex in examples:
            part_data_test.setdefault(ex[3],[]).append(ex) 
        self.assertEqual(part_data_test,part_data) 
           
    def test_discrete_predict(self): 
        train_data, test_data = load_project_data('example')
        examples = [ex for ex in train_data] + [ex for ex in test_data]
        data = ExampleSet(examples)    
        
        n = Node()
        
        n.train(data,[1,3]) #only train on the discrete data
        
        self.assertTrue(all([ex[-1]==n.predict(ex) for ex in data]))
        
        n.train(data,[3,1]) #only train on the discrete data
        
        self.assertTrue(all([ex[-1]==n.predict(ex) for ex in data]))

            
          

if __name__ == "__main__": 
    unittest.main()